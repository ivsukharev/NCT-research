// NctCli.cs
// Unified train/infer CLI для NCT
// Использование:
//   dotnet run -- train --data data/train.csv --output model/model.bin --config model/meta.json
//   dotnet run -- infer --model model/model.bin --input data/test.csv --output pred.json

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NCT_framework;

namespace NctCli
{
public class NctCliProgram
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
            return;


        string mode = args[0];

        try
        {
                if (mode == "train")
                    RunTrain(args.Skip(1).ToArray());
                else if (mode == "infer")
                    RunInfer(args.Skip(1).ToArray());
                else
                    return;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] {ex.Message}");
            Environment.Exit(1);
        }
    }

    static void RunTrain(string[] args)
    {
        string dataCsv = null, outputPath = null, configJson = null;
        int totalClasses = 60, ownClasses = 10, featuresCount = 512, imgPerClass = 14;
        int neuronsNumber = 128, neuronsInputsNumber = 4;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--data" && i + 1 < args.Length) dataCsv = args[++i];
            else if (args[i] == "--output" && i + 1 < args.Length) outputPath = args[++i];
            else if (args[i] == "--classes" && i + 1 < args.Length) totalClasses = int.Parse(args[++i]);
            else if (args[i] == "--own-classes" && i + 1 < args.Length) ownClasses = int.Parse(args[++i]);
        }

        if (string.IsNullOrEmpty(dataCsv) || string.IsNullOrEmpty(outputPath))
            throw new ArgumentException("Missing required arguments: --data, --output");

        Console.WriteLine($"[TRAIN] Reading {dataCsv}...");
        var dataOwns = DataFactory.ExtractFeaturesFromFile(dataCsv, totalClasses, imgPerClass, featuresCount);
        dataOwns = dataOwns.Take(totalClasses).ToList();


        var random = new Random(42);
        var ncts = new NCT[ownClasses];
        var keys = new BitArray[ownClasses];
        var sxStrangers = new double[ownClasses][]; // удалить?
        var trainingMetrics = new List<TrainMetrics>();

        Console.WriteLine($"[TRAIN] Training {ownClasses} NCTs...");

        for (int i = 0; i < ownClasses; i++)
        {
            Console.Write($"  [NCT {i}/{ownClasses}] ");

            // Выбираем 9 "Своих" примеров
            var owns = new List<double[]>();
            for (int k = 0; k < 9; k++)
            {
                int imgIndex = random.Next(0, dataOwns[i].Count);
                owns.Add(dataOwns[i][imgIndex]);
                dataOwns[i].RemoveAt(imgIndex);
            }

            // Выбираем "Чужих" из других классов
            var strangers = new List<double[]>();
            for (int k = 0; k < totalClasses; k++)
            {
                if (i != k && k > ownClasses - 1)
                {
                    int index = random.Next(0, dataOwns[k].Count);
                    strangers.Add(dataOwns[k][index]);
                    dataOwns[k].RemoveAt(index);
                }
            }

            // Генерируем ключ и обучаем NCT
            keys[i] = KeyFactory.GetKey(random, 128);
            ncts[i] = new NCT();
            ncts[i].Training(owns, strangers, featuresCount, keys[i], neuronsNumber, neuronsInputsNumber, random);

            // Валидация NCT
            var ownTest = dataOwns[i].ToList();
            var strangerTest = new List<double[]>();

            for (int k = 0; k < totalClasses; k++)
                if (i != k && k > ownClasses - 1 && dataOwns[k].Count > 0)
                    strangerTest.Add(dataOwns[k][0]);

            var metrics = EvaluateNctQuality(ncts[i], keys[i], ownTest, strangerTest, i);

            trainingMetrics.Add(metrics);
        }

        // Сохраняем модель
        Console.WriteLine($"[TRAIN] Saving model to {outputPath}...");
        SaveModelAsJson(ncts, keys, outputPath, ownClasses, totalClasses, featuresCount, neuronsNumber, neuronsInputsNumber);

        Console.WriteLine($"[EVAL] Evaluating training quality...");
        PrintTrainingSummary(trainingMetrics);

        Console.WriteLine("[DONE] Training complete!");
    }


    static void RunInfer(string[] args)
    {
        string modelPath = null, inputCsv = null, outputJson = null;
        int targetNct = -1;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length) modelPath = args[++i];
            else if (args[i] == "--input" && i + 1 < args.Length) inputCsv = args[++i];
            else if (args[i] == "--output" && i + 1 < args.Length) outputJson = args[++i];
            else if (args[i] == "--target-nct" && i + 1 < args.Length) targetNct = int.Parse(args[++i]);
        }

        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(inputCsv) || string.IsNullOrEmpty(outputJson))
            throw new ArgumentException("Missing required arguments: --model, --input, --output");

        // Загружаем модель
        Console.WriteLine($"[INFER] Loading model from {modelPath}...");
        var (ncts, keys, meta) = LoadModelFromJson(modelPath);

        // Читаем входной CSV
        Console.WriteLine($"[INFER] Reading input from {inputCsv}...");
        int featureCount = Convert.ToInt32(meta.feature_count);
        var testData = ReadInputCsv(inputCsv, featureCount);

        // Выполняем инференс
        Console.WriteLine($"[INFER] Running inference on {testData.Count} samples...");
        var predictions = new List<object>();

        for (int sampleIdx = 0; sampleIdx < testData.Count; sampleIdx++)
        {
            int id = testData[sampleIdx].Item1;
            int trueClass = testData[sampleIdx].Item2;
            double[] features = testData[sampleIdx].Item3;

            var code = ncts[targetNct].VerifyImage(features);
            int hamming = ComputeHamming(code, keys[targetNct]);

            predictions.Add(new
            {
                id = id,
                true_class = trueClass,
                target_nct = targetNct,
                hamming_distance = hamming,
                bit_code = BitArrayToString(code)
            });

            if ((sampleIdx + 1) % 100 == 0)
                Console.WriteLine($"  Processed {sampleIdx + 1}/{testData.Count}");
        }

        // Сохраняем результаты
        Console.WriteLine($"[INFER] Saving predictions to {outputJson}...");
        var result = new
        {
            model_version = meta.version,
            feature_count = meta.feature_count,
            own_classes = meta.own_classes,
            timestamp = DateTime.UtcNow.ToString("O"),
            predictions = predictions
        };

        File.WriteAllText(outputJson, JsonConvert.SerializeObject(result, Formatting.Indented));
        Console.WriteLine("[✓] Inference complete!");
    }

    // ========== UTILITY FUNCTIONS ==========

    static TrainMetrics EvaluateNctQuality(NCT nct, BitArray key, List<double[]> owns, List<double[]> strangers, int nctIndex)
    {
        var metrics = new TrainMetrics { NctIndex = nctIndex };

        // Оцениваем на обучающих примерах "Свой"
        var ownHamming = new List<int>();
        foreach (var ownFeatures in owns)
        {
            BitArray code = nct.VerifyImage(ownFeatures);
            int hamming = ComputeHamming(code, key);
            ownHamming.Add(hamming);
        }

        // Оцениваем на обучающих примерах "Чужой"
        var strangerHamming = new List<int>();
        foreach (var strangerFeatures in strangers)
        {
            BitArray code = nct.VerifyImage(strangerFeatures);
            int hamming = ComputeHamming(code, key);
            strangerHamming.Add(hamming);
        }

        // Вычисляем статистику
        metrics.OwnHammingMean = ownHamming.Average();
        metrics.OwnHammingStd = ComputeStdDev(ownHamming);
        metrics.OwnHammingMin = ownHamming.Min();
        metrics.OwnHammingMax = ownHamming.Max();

        metrics.StrangerHammingMean = strangerHamming.Average();
        metrics.StrangerHammingStd = ComputeStdDev(strangerHamming);
        metrics.StrangerHammingMin = strangerHamming.Min();
        metrics.StrangerHammingMax = strangerHamming.Max();

        int threshold = 15;
        int correctOwns = ownHamming.Count(h => h < threshold);
        int correctStrangers = strangerHamming.Count(h => h >= threshold);

        metrics.TrainPrecision = (double)correctOwns / owns.Count;
        metrics.TrainRecall = (double)correctStrangers / strangers.Count;
        metrics.TrainAccuracy = (correctOwns + correctStrangers) / (double)(owns.Count + strangers.Count);

        return metrics;
    }
    // ====================================================================
    //                              DEBUG
    // ====================================================================

    // ====================================================================
    //                              DEBUG
    // ====================================================================


    // ========== Save to JSON ==========
    static void SaveModelAsJson(NCT[] ncts, BitArray[] keys, string filepath,
            int ownClasses, int totalClasses, int featureCount, int neuronsNumber, int neuronsInputsNumber)
    {

        var nctsList = new List<dynamic>();
        for (int i = 0; i < ncts.Length; i++)
            nctsList.Add(SerializeNCTToJson(ncts[i], i, keys[i]));

        var model = new
        {
            version = 1,
            timestamp = DateTime.UtcNow.ToString("O"),
            own_classes = ownClasses,
            total_classes = totalClasses,
            feature_count = featureCount,
            neurons_count = neuronsNumber,
            neurons_input_count = neuronsInputsNumber,
            ncts = nctsList
        };

        string json = JsonConvert.SerializeObject(model, Formatting.Indented);
        File.WriteAllText(filepath, json);
    }

    static dynamic SerializeNCTToJson(NCT nct, int id, BitArray key)
    {

        var weights = GetField<List<double[]>>(nct, "_w");
        var thresholds = GetField<List<double[]>>(nct, "_thresholds");
        var tableIndices = GetField<List<int>>(nct, "_tablesIndexes");
        var sxstranger = GetField<double[]>(nct, "_sx_stranger");
        var synapses = ExtractSynapsesFromNCT(nct);

        return new
        {
            id = id,
            weights = weights?.ToArray() ?? new double[0][],
            thresholds = thresholds?.ToArray() ?? new double[0][],
            table_indices = tableIndices?.ToArray() ?? new int[0],
            sx_stranger = sxstranger ?? new double[0],
            synapses = synapses,
            key_bits = BitArrayToString(key),
            serialized_at = DateTime.UtcNow.ToString("O")
        };
    }


    // загрузить из JSON
    public static (NCT[] ncts, BitArray[] keys, dynamic meta) LoadModelFromJson(string filepath)
    {

        if (!File.Exists(filepath))
            throw new FileNotFoundException($"Model file not found: {filepath}");

        string json = File.ReadAllText(filepath);
        dynamic model = JsonConvert.DeserializeObject(json);

        int nctCount = model.ncts.Count;
        var ncts = new NCT[nctCount];
        var keys = new BitArray[nctCount];

        Console.WriteLine($"  Found {nctCount} NCTs");
        Console.WriteLine($"  Features: {model.feature_count}, Neurons: {model.neurons_count}");

        for (int i = 0; i < nctCount; i++)
        {
            ncts[i] = DeserializeNCTFromJson(model.ncts[i]);
            keys[i] = StringToBitArray(model.ncts[i].key_bits.ToString());
        }

        return (ncts, keys, model);
    }

    static NCT DeserializeNCTFromJson(dynamic nctJson)
    {
        try
        {
            var nct = new NCT();
            
            // Веса (List<double[]>)
            if (nctJson.weights != null)
            {
                var weightsArray = JsonConvert.DeserializeObject<double[][]>(
                    nctJson.weights.ToString()) ?? Array.Empty<double[]>();
                var weightsList = new List<double[]>(weightsArray);
                SetField(nct, "_w", weightsList);
            }

            // Пороги (List<double[]>)
            if (nctJson.thresholds != null)
            {
                var thresholdsArray = JsonConvert.DeserializeObject<double[][]>(
                    nctJson.thresholds.ToString()) ?? Array.Empty<double[]>();
                var thresholdsList = new List<double[]>(thresholdsArray);
                SetField(nct, "_thresholds", thresholdsList);
            }

            // Индексы таблиц (проверяем оба варианта имени поля)
            if (nctJson.table_indices != null)
            {
                var tableIndexesArray = JsonConvert.DeserializeObject<int[]>(
                    nctJson.table_indices.ToString()) ?? Array.Empty<int>();
                var tableIndexesList = new List<int>(tableIndexesArray);
                SetField(nct, "_tablesIndexes", tableIndexesList);
            }
            else if (nctJson.table_indexes != null)
            {
                var tableIndexesArray = JsonConvert.DeserializeObject<int[]>(
                    nctJson.table_indexes.ToString()) ?? Array.Empty<int>();
                var tableIndexesList = new List<int>(tableIndexesArray);
                SetField(nct, "_tablesIndexes", tableIndexesList);
            }

            // sx_stranger
            if (nctJson.sx_stranger != null)
            {
                var sxstranger = JsonConvert.DeserializeObject<double[]>(
                    nctJson.sx_stranger.ToString());
                SetField(nct, "_sx_stranger", sxstranger);
            }

            // Синапсы (преобразуем List<List<int[]>> обратно в List<MetaFeature[]>)
            if (nctJson.synapses != null)
            {
                var synapsesData = JsonConvert.DeserializeObject<List<List<int[]>>>(
                    nctJson.synapses.ToString());
                if (synapsesData != null)
                {
                    var synapsesList = new List<MetaFeature[]>();
                    foreach (var neuronSynapses in synapsesData)
                    {
                        if (neuronSynapses != null)
                        {
                            var metaFeatures = new MetaFeature[neuronSynapses.Count];
                            for (int i = 0; i < neuronSynapses.Count; i++)
                            {
                                if (neuronSynapses[i] != null && neuronSynapses[i].Length >= 2)
                                {
                                    metaFeatures[i].j = neuronSynapses[i][0];
                                    metaFeatures[i].t = neuronSynapses[i][1];
                                }
                            }
                            synapsesList.Add(metaFeatures);
                        }
                    }
                    SetField(nct, "_synapses", synapsesList);
                }
            }

            return nct;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to deserialize NCT: {ex.Message}");
            throw;
        }
    }


    // Extract synapses
    static List<List<int[]>> ExtractSynapsesFromNCT(NCT nct)
    {
        var result = new List<List<int[]>>();
        var synapsesObj = GetField<List<MetaFeature[]>>(nct, "_synapses");
        foreach (var neuronMetaFeatures in synapsesObj)
        {
            var neuronSynapses = new List<int[]>();

            if (neuronMetaFeatures != null)
            {
                foreach (var metaFeature in neuronMetaFeatures)
                {
                    int[] synapseArray = new int[] { metaFeature.j, metaFeature.t };
                    neuronSynapses.Add(synapseArray);
                }
            }
            result.Add(neuronSynapses);
        }

        return result;
    }


    // ========== HELPERS ==========

    static T GetField<T>(object obj, string fieldName) where T : class
    {
        var field = obj.GetType().GetField(fieldName,
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.IgnoreCase);

        // return (T)field.GetValue(obj);
        return field?.GetValue(obj) as T;
    }

    static void SetField<T>(object obj, string fieldName, T value) where T : class
    {
        var field = obj.GetType().GetField(fieldName,
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.IgnoreCase);

        field?.SetValue(obj, value);
    }

    static int[] ConvertToIntArray(object obj)
    {
        if (obj is int[] arr) return arr;

        if (obj is System.Collections.IList list && list.Count >= 2)
        {
            var result = new int[2];
            if (int.TryParse(list[0].ToString(), out int i))
                result[0] = i;
            if (int.TryParse(list[1].ToString(), out int t))
                result[1] = t;
            return result;
        }

        // Попробуем ValueTuple(int, int)
        try
        {
            var type = obj?.GetType();
            if (type?.Name == "ValueTuple`2")
            {
                var item1 = (int)type.GetProperty("Item1")?.GetValue(obj);
                var item2 = (int)type.GetProperty("Item2")?.GetValue(obj);
                return new int[] { item1, item2 };
            }
        }
        catch { }

        return null;
    }
    static List<(int id, int classLabel, double[] features)> ReadInputCsv(string csvPath, int featureCount)
    {
        var data = new List<(int, int, double[])>();
        var lines = File.ReadAllLines(csvPath);

        // Пропускаем заголовок
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 3 + featureCount) continue;

            int id = int.Parse(parts[0]);
            int classLabel = int.Parse(parts[1]);

            var features = new double[featureCount];
            for (int j = 0; j < featureCount; j++)
                features[j] = double.Parse(parts[3 + j], System.Globalization.CultureInfo.InvariantCulture);

            data.Add((id, classLabel, features));
        }

        return data;
    }

    // Вспомогательные методы

    static void PrintTrainingSummary(List<TrainMetrics> allMetrics)
    {

        double avgOwnHamming = allMetrics.Average(m => m.OwnHammingMean);
        double avgStrangerHamming = allMetrics.Average(m => m.StrangerHammingMean);
        double avgAccuracy = allMetrics.Average(m => m.TrainAccuracy);

        Console.WriteLine($"Across all {allMetrics.Count} NCTs:");
        Console.WriteLine($"  Avg Own Hamming:      {avgOwnHamming:F2}");
        Console.WriteLine($"  Avg Stranger Hamming: {avgStrangerHamming:F2}");
        Console.WriteLine($"  Avg Train Accuracy:   {avgAccuracy:P2}");
    }

    static int ComputeHamming(BitArray code, BitArray key)
    {
        int hamming = 0;
        for (int i = 0; i < code.Length && i < key.Length; i++)
            if (code[i] != key[i])
                hamming++;
        return hamming;
    }

    static double ComputeStdDev(List<int> values)
    {
        if (values.Count <= 1) return 0;
        double mean = values.Average();
        double sumSquares = values.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquares / values.Count);
    }

    static string BitArrayToString(BitArray ba)
    {
        var sb = new System.Text.StringBuilder(ba.Length);
        foreach (bool bit in ba)
            sb.Append(bit ? '1' : '0');
        return sb.ToString();
    }
    static BitArray StringToBitArray(string str)
    {
        var bitArray = new BitArray(str.Length);
        for (int i = 0; i < str.Length; i++)
            bitArray[i] = (str[i] == '1');
        return bitArray;
    }

    static void SerializeBitArray(BinaryWriter writer, BitArray ba)
    {
        writer.Write(ba.Length);
        byte[] bytes = new byte[(ba.Length + 7) / 8];
        ba.CopyTo(bytes, 0);
        writer.Write(bytes);
    }

    static BitArray DeserializeBitArray(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        byte[] bytes = reader.ReadBytes((length + 7) / 8);
        return new BitArray(bytes) { Length = length };
    }

    public class TrainMetrics
    {
        [JsonProperty("nct_index")]
        public int NctIndex { get; set; }

        [JsonProperty("own_hamming_mean")]
        public double OwnHammingMean { get; set; }

        [JsonProperty("own_hamming_std")]
        public double OwnHammingStd { get; set; }

        [JsonProperty("own_hamming_min")]
        public int OwnHammingMin { get; set; }

        [JsonProperty("own_hamming_max")]
        public int OwnHammingMax { get; set; }

        [JsonProperty("stranger_hamming_mean")]
        public double StrangerHammingMean { get; set; }

        [JsonProperty("stranger_hamming_std")]
        public double StrangerHammingStd { get; set; }

        [JsonProperty("stranger_hamming_min")]
        public int StrangerHammingMin { get; set; }

        [JsonProperty("stranger_hamming_max")]
        public int StrangerHammingMax { get; set; }

        [JsonProperty("hamming_margin")]
        public double HammingMargin { get; set; }

        [JsonProperty("train_precision")]
        public double TrainPrecision { get; set; }

        [JsonProperty("train_recall")]
        public double TrainRecall { get; set; }

        [JsonProperty("train_accuracy")]
        public double TrainAccuracy { get; set; }
    }
}
}