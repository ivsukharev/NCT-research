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
using NCT_framework;

class NctCliProgram
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return;
        }

        string mode = args[0];

        try
        {
            if (mode == "train")
                RunTrain(args.Skip(1).ToArray());
            else if (mode == "infer")
                RunInfer(args.Skip(1).ToArray());
            else
                PrintUsage();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] {ex.Message}");
            Environment.Exit(1);
        }
    }

    static void PrintUsage()
    {
        Console.WriteLine(@"
Usage:
  dotnet run -- train --data <csv> --output <bin> --config <json> [--classes N] [--own-classes M]
  dotnet run -- infer --model <bin> --meta <json> --input <csv> --output <json>

Examples:
  dotnet run -- train --data data/train.csv --output model/model.bin --config model/meta.json --classes 40 --own-classes 10
  dotnet run -- infer --model model/model.bin --meta model/meta.json --input data/test.csv --output pred.json
");
    }

    // ========== TRAIN MODE ==========
    static void RunTrain(string[] args)
    {
        string dataCsv = null, outputBin = null, configJson = null;
        int totalClasses = 40, ownClasses = 10;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--data" && i + 1 < args.Length) dataCsv = args[++i];
            else if (args[i] == "--output" && i + 1 < args.Length) outputBin = args[++i];
            else if (args[i] == "--config" && i + 1 < args.Length) configJson = args[++i];
            else if (args[i] == "--classes" && i + 1 < args.Length) totalClasses = int.Parse(args[++i]);
            else if (args[i] == "--own-classes" && i + 1 < args.Length) ownClasses = int.Parse(args[++i]);
        }

        if (string.IsNullOrEmpty(dataCsv) || string.IsNullOrEmpty(outputBin) || string.IsNullOrEmpty(configJson))
            throw new ArgumentException("Missing required arguments: --data, --output, --config");

        Console.WriteLine($"[TRAIN] Reading {dataCsv}...");
        var dataOwns = DataFactory.ExtractFeaturesFromFile(dataCsv, totalClasses, 14, 512);
        dataOwns = dataOwns.Take(totalClasses).ToList();

        int featuresCount = 512;
        var random = new Random(42);

        // Обучение NCT для каждого класса
        var ncts = new NCT[ownClasses];
        var keys = new BitArray[ownClasses];
        var sxStrangers = new double[ownClasses][]; // Сохраняем для каждого NCT

        Console.WriteLine($"[TRAIN] Training {ownClasses} NCTs...");

        for (int i = 0; i < ownClasses; i++)
        {
            Console.WriteLine($"  [NCT {i}/{ownClasses}]");

            // Выбираем 10 "Своих" примеров
            var owns = new List<double[]>();
            for (int k = 0; k < 10; k++)
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
                    if (index > 0)
                    {
                        strangers.Add(dataOwns[k][index]);
                        dataOwns[k].RemoveAt(index);
                    }
                }
            }

            // Генерируем ключ и обучаем NCT
            keys[i] = KeyFactory.GetKey(random, 128);
            ncts[i] = new NCT();
            ncts[i].Training(owns, strangers, featuresCount, keys[i], 128, 4, random, i);
        }

        // Сохраняем модель
        Console.WriteLine($"[TRAIN] Saving model to {outputBin}...");
        SaveModel(ncts, keys, outputBin);

        // Сохраняем метаданные
        var meta = new
        {
            version = 1,
            feature_count = 512,
            total_classes = totalClasses,
            own_classes = ownClasses,
            bits_per_neuron = 2,
            timestamp = DateTime.UtcNow.ToString("O")
        };

        Console.WriteLine($"[TRAIN] Saving metadata to {configJson}...");
        File.WriteAllText(configJson, JsonConvert.SerializeObject(meta, Formatting.Indented));

        Console.WriteLine("[✓] Training complete!");
    }

    // ========== INFER MODE ==========
    static void RunInfer(string[] args)
    {
        string modelBin = null, metaJson = null, inputCsv = null, outputJson = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length) modelBin = args[++i];
            else if (args[i] == "--meta" && i + 1 < args.Length) metaJson = args[++i];
            else if (args[i] == "--input" && i + 1 < args.Length) inputCsv = args[++i];
            else if (args[i] == "--output" && i + 1 < args.Length) outputJson = args[++i];
        }

        if (string.IsNullOrEmpty(modelBin) || string.IsNullOrEmpty(inputCsv) || string.IsNullOrEmpty(outputJson))
            throw new ArgumentException("Missing required arguments: --model, --input, --output");

        // Загружаем модель
        Console.WriteLine($"[INFER] Loading model from {modelBin}...");
        var (ncts, keys, meta) = LoadModel(modelBin, metaJson);

        // Читаем входной CSV
        Console.WriteLine($"[INFER] Reading input from {inputCsv}...");
        var testData = ReadInputCsv(inputCsv, meta.feature_count);

        // Выполняем инференс
        Console.WriteLine($"[INFER] Running inference on {testData.Count} samples...");
        var predictions = new List<object>();

        for (int sampleIdx = 0; sampleIdx < testData.Count; sampleIdx++)
        {
            var (id, classLabel, features) = testData[sampleIdx];

            int bestClass = -1;
            int bestHamming = int.MaxValue;
            string bestBitArray = "";

            // Проверяем через каждый NCT и выбираем класс с минимальным Хэммингом
            for (int nctIdx = 0; nctIdx < ncts.Length; nctIdx++)
            {
                var code = ncts[nctIdx].VerifyImage(features);
                int hamming = ComputeHamming(code, keys[nctIdx]);

                if (hamming < bestHamming)
                {
                    bestHamming = hamming;
                    bestClass = nctIdx;
                    bestBitArray = BitArrayToString(code);
                }
            }

            predictions.Add(new
            {
                id = id,
                true_class = classLabel,
                pred_class = bestClass,
                best_hamming = bestHamming,
                bit_array = bestBitArray
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

    static void SaveModel(NCT[] ncts, BitArray[] keys, string filepath)
    {
        using (var writer = new BinaryWriter(File.Create(filepath)))
        {
            // Magic + version
            writer.Write("NCTM".ToCharArray()); // magic
            writer.Write(1); // version

            // Размеры
            writer.Write(ncts.Length); // количество NCT
            writer.Write(512); // featuresCount (зафиксировано)

            // Сохраняем каждый NCT
            for (int i = 0; i < ncts.Length; i++)
            {
                Console.WriteLine($"    Saving NCT {i}...");
                ncts[i].SerializeToWriter(writer);
                SerializeBitArray(writer, keys[i]);
            }
        }
    }

    static (NCT[] ncts, BitArray[] keys, dynamic meta) LoadModel(string modelBin, string metaJson)
    {
        dynamic meta = null;
        if (!string.IsNullOrEmpty(metaJson))
            meta = JsonConvert.DeserializeObject(File.ReadAllText(metaJson));

        using (var reader = new BinaryReader(File.OpenRead(modelBin)))
        {
            // Magic + version
            string magic = new string(reader.ReadChars(4));
            if (magic != "NCTM")
                throw new Exception("Invalid model file format");

            int version = reader.ReadInt32();
            if (version != 1)
                throw new Exception($"Unsupported version: {version}");

            // Размеры
            int nctCount = reader.ReadInt32();
            int featuresCount = reader.ReadInt32();

            var ncts = new NCT[nctCount];
            var keys = new BitArray[nctCount];

            for (int i = 0; i < nctCount; i++)
            {
                ncts[i] = NCT.DeserializeFromReader(reader);
                keys[i] = DeserializeBitArray(reader);
            }

            return (ncts, keys, meta);
        }
    }

    static List<(int id, int classLabel, double[] features)> ReadInputCsv(string csvPath, int featureCount)
    {
        var data = new List<(int, int, double[])>();
        var lines = File.ReadAllLines(csvPath);

        // Пропускаем заголовок
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            int id = int.Parse(parts[0]);
            int classLabel = int.Parse(parts[1]);
            // parts[2] = split (пропускаем)

            var features = new double[featureCount];
            for (int j = 0; j < featureCount; j++)
                features[j] = double.Parse(parts[3 + j]);

            data.Add((id, classLabel, features));
        }

        return data;
    }

    static int ComputeHamming(BitArray code, BitArray key)
    {
        int hamming = 0;
        for (int i = 0; i < code.Length && i < key.Length; i++)
            if (code[i] != key[i])
                hamming++;
        return hamming;
    }

    static string BitArrayToString(BitArray ba)
    {
        var sb = new System.Text.StringBuilder(ba.Length);
        foreach (bool bit in ba)
            sb.Append(bit ? '1' : '0');
        return sb.ToString();
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
}