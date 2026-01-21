using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Collections;
using System.Globalization;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NCT_framework;
using NctCli;


public class GraphFeature
{
    [JsonProperty("importance")]
    public double Importance { get; set; }
    
    [JsonProperty("partners")]
    public Dictionary<string, double> Partners { get; set; } = new Dictionary<string, double>();
}

public class CorrelationGraph
{
    [JsonProperty("features")]
    public Dictionary<string, GraphFeature> Features { get; set; } = new Dictionary<string, GraphFeature>();
}

public class ConstrainedOptimizerGraph
{
    private readonly NCT nct;
    private readonly BitArray key;
    private readonly CorrelationGraph graph;
    
    private readonly int targetNct;
    private readonly double learningRate;
    private readonly double stepSize;
    private readonly bool earlyStop;
    private readonly int earlyStopping;
    
    private readonly double importanceThreshold;
    private readonly List<string> parentFeatures; // родительские признаки (importance >= порог)
    
    public ConstrainedOptimizerGraph(
        NCT nct,
        BitArray key,
        string graphJsonPath,
        int targetNct = 0,
        double learningRate = 0.01,
        double stepSize = 1.0,
        int earlyStopping = 30,
        bool earlyStop = false
    )
    {
        this.nct = nct;
        this.key = key;
        this.targetNct = targetNct;
        this.learningRate = learningRate;
        this.stepSize = stepSize;
        if (earlyStopping > 0)
        {
            this.earlyStop = true;
            this.earlyStopping = earlyStopping;
        }
        
        this.graph = LoadGraph(graphJsonPath);
        this.importanceThreshold = ComputeImportanceThreshold();
        this.parentFeatures = GetParentFeatures();

        Console.WriteLine($"[DONE] Граф загружен");
        Console.WriteLine($"  - Порог importance: {importanceThreshold:F4}");
        Console.WriteLine($"  - Признаков с importance >= порога: {parentFeatures.Count}");
    }
    
    // Загрузить граф из JSON файла
    private CorrelationGraph LoadGraph(string graphJsonPath)
    {
        string json = File.ReadAllText(graphJsonPath);
        return JsonConvert.DeserializeObject<CorrelationGraph>(json); 
    }
    
    /// Вычислить порог importance как среднее ненулевых значений
    private double ComputeImportanceThreshold()
    {
        var nonZeroImportance = graph.Features.Values
            .Select(f => f.Importance)
            .Where(imp => imp > 0.0)
            .ToList();
        
        double threshold = nonZeroImportance.Average();
        return threshold;
    }
    
    /// Получить список родительских признаков (importance >= порог)
    private List<string> GetParentFeatures()
    {
        var parents = graph.Features
            .Where(kvp => kvp.Value.Importance >= importanceThreshold)
            .OrderByDescending(kvp => kvp.Value.Importance)
            .Select(kvp => kvp.Key)
            .ToList();
        
        return parents;
    }
    
    /// <summary>
    /// Выполнить атаку на образец
    /// </summary>
    public (double[] adversarialImage, AttackMetrics metrics) Attack(
        double[] image,
        int trueClass,
        int nIterations = 100,
        bool verbose = true
    )
    {
        double[] currentImage = (double[])image.Clone();
        var distancesHistory = new List<int>();
        int bestDistance = int.MaxValue;
        int patienceCounter = 0;
        
        for (int iteration = 0; iteration < nIterations; iteration++)
        {
            int currentDistance = ComputeHammingDistance(currentImage, trueClass);
            distancesHistory.Add(currentDistance);
            
            // Логирование прогресса
            if (verbose && (iteration % 10 == 0 || iteration == 0))
            {
                int delta = distancesHistory[0] - currentDistance;
                Console.WriteLine($"  Итерация {iteration,4:D}: Hamming distance = {currentDistance,4:D} (улучшение: {delta,4:D})");
            }
            
            //  Early stopping
            if (earlyStop)
            {
                if (currentDistance < bestDistance)
                {
                    bestDistance = currentDistance;
                    patienceCounter = 0;
                }
                else
                {
                    patienceCounter++;
                }
                
                if (patienceCounter >= earlyStopping)
                {
                    Console.WriteLine($"Early stopping: {earlyStopping} итераций без улучшения");
                    return (currentImage, BuildMetrics(
                        distancesHistory,
                        iteration,
                        earlyStoppedFlag: true,
                        reason: "No improvement for 10 iterations"
                    ));
                }
            }
            
            //  Для каждого родительского признака
            foreach (string parentId in parentFeatures)
            {
                GraphFeature parentFeat = graph.Features[parentId];
                Dictionary<string, double> partners = parentFeat.Partners;
                
                // ля каждого дочернего признака (партнёра)
                foreach (var kvp in partners)
                {
                    string partnerIdStr = kvp.Key;
                    double partnerImportance = kvp.Value;
                    
                    if (!int.TryParse(partnerIdStr, out int partnerId))
                        continue;
                    
                    if (partnerId < 0 || partnerId >= currentImage.Length)
                        continue;
                    
                    // максимальное изменение
                    double maxChange = (1.0 - partnerImportance) * learningRate * stepSize;
                    
                    // градиент
                    double gradient = ComputeGradient(partnerId, trueClass, currentImage);
                    
                    if (gradient > 0)
                        currentImage[partnerId] -= maxChange;
                    else
                        currentImage[partnerId] += maxChange;
                }
            }
        }
        
        // Console.WriteLine("[DONE] Атака завершена");
        // Console.WriteLine($"  - Исходное расстояние: {distancesHistory[0]}");
        // Console.WriteLine($"  - Финальное расстояние: {distancesHistory[distancesHistory.Count - 1]}");
        // Console.WriteLine($"  - Улучшение: {distancesHistory[0] - distancesHistory[distancesHistory.Count - 1]}");
        // Console.WriteLine($"  - Итераций выполнено: {nIterations}");
        
        return (currentImage, BuildMetrics(
            distancesHistory,
            nIterations,
            earlyStoppedFlag: false,
            reason: "Max iterations reached"
        ));
    }
    
    /// <summary>
    /// Вычислить расстояние Хэмминга
    /// </summary>
    private int ComputeHammingDistance(double[] image, int trueClass)
    {
        BitArray code = nct.VerifyImage(image);
        BitArray keyBits = key;
        
        int distance = 0;
        int minLength = Math.Min(code.Count, keyBits.Count);
        
        for (int i = 0; i < minLength; i++)
        {
            if (code[i] != keyBits[i])
                distance++;
        }
        
        return distance;
    }
    
    /// <summary>
    /// Вычислить эмпирический градиент по одному признаку
    /// </summary>
    private double ComputeGradient(int featureId, int trueClass, double[] image)
    {
        double epsilon = 0.01;
        
        int distanceOriginal = ComputeHammingDistance(image, trueClass);
        
        double[] imagePerturbated = (double[])image.Clone();
        imagePerturbated[featureId] += epsilon;
        
        int distancePerturbated = ComputeHammingDistance(imagePerturbated, trueClass);
        
        double gradient = (distancePerturbated - distanceOriginal) / epsilon;
        return gradient;
    }
    
    /// <summary>
    /// Собрать метрики
    /// </summary>
    private AttackMetrics BuildMetrics(
        List<int> distancesHistory,
        int iterationsCompleted,
        bool earlyStoppedFlag,
        string reason
    )
    {
        return new AttackMetrics
        {
            InitialHammingDistance = distancesHistory[0],
            FinalHammingDistance = distancesHistory[distancesHistory.Count - 1],
            Improvement = distancesHistory[0] - distancesHistory[distancesHistory.Count - 1],
            IterationsCompleted = iterationsCompleted,
            DistancesHistory = distancesHistory,
            StoppedEarly = earlyStoppedFlag,
            Reason = reason,
            LearningRate = learningRate,
            StepSize = stepSize
        };
    }
}

/// <summary>
/// Метрики одной атаки
/// </summary>
public class AttackMetrics
{
    [JsonProperty("initial_hamming_distance")]
    public int InitialHammingDistance { get; set; }
    
    [JsonProperty("final_hamming_distance")]
    public int FinalHammingDistance { get; set; }
    
    [JsonProperty("improvement")]
    public int Improvement { get; set; }
    
    [JsonProperty("iterations_completed")]
    public int IterationsCompleted { get; set; }
    
    [JsonProperty("distances_history")]
    public List<int> DistancesHistory { get; set; }
    
    [JsonProperty("stopped_early")]
    public bool StoppedEarly { get; set; }
    
    [JsonProperty("reason")]
    public string Reason { get; set; }
    
    [JsonProperty("learning_rate")]
    public double LearningRate { get; set; }
    
    [JsonProperty("step_size")]
    public double StepSize { get; set; }
    
    [JsonProperty("sample_index")]
    public int SampleIndex { get; set; }
}

/// <summary>
/// Главный класс для управления атакой
/// </summary>
public class AttackRunnerGraph
{
    public static void Main(string[] args)
    {
        string graphJsonPath = null;
        string modelPath = null;
        string inputCsv = null;
        string outputDir = null;
        double learningRate = 0.01;
        double stepSize = 1.0;
        int nIterations = 100;
        int batchSize = 10;
        int targetNct = 0;
        int earlyStopping = 30;
        
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--graph-json" && i + 1 < args.Length)
                graphJsonPath = args[++i];
            else if (args[i] == "--model" && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i] == "--input" && i + 1 < args.Length)
                inputCsv = args[++i];
            else if (args[i] == "--output" && i + 1 < args.Length)
                outputDir = args[++i];
            else if (args[i] == "--learning-rate" && i + 1 < args.Length)
                learningRate = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--step-size" && i + 1 < args.Length)
                stepSize = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--n-iterations" && i + 1 < args.Length)
                nIterations = int.Parse(args[++i]);
            else if (args[i] == "--early-stopping" && i + 1 < args.Length)
                earlyStopping = int.Parse(args[++i]);
            else if (args[i] == "--batch-size" && i + 1 < args.Length)
                batchSize = int.Parse(args[++i]);
            else if (args[i] == "--target-nct" && i + 1 < args.Length)
                targetNct = int.Parse(args[++i]);
            else if (args[i] == "--help")
            {
                PrintHelp();
                return;
            }
        }
        
        if (string.IsNullOrEmpty(graphJsonPath) || string.IsNullOrEmpty(modelPath) || 
            string.IsNullOrEmpty(inputCsv) || string.IsNullOrEmpty(outputDir))
        {
            Console.WriteLine("Error: Missing required arguments");
            PrintHelp();
            return;
        }
        
        try
        {
            if (!File.Exists(graphJsonPath))
                throw new FileNotFoundException($"Graph JSON not found: {graphJsonPath}");
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model not found: {modelPath}");
            if (!File.Exists(inputCsv))
                throw new FileNotFoundException($"Input CSV not found: {inputCsv}");
            
            
            var (ncts, keys, meta) = NctCliProgram.LoadModelFromJson(modelPath);
            Console.WriteLine($"[DONE] Модель загружена: {modelPath}");
            Console.WriteLine($"  - Feature count: {meta.feature_count}");
            Console.WriteLine($"  - NCTs: {ncts.Length}");
            Console.WriteLine("");
            
            int featureCount = Convert.ToInt32(meta.feature_count);
            var data = LoadDataFromCsv(inputCsv, featureCount);
            Console.WriteLine($"[DONE] Данные из CSV загружены: {data.Count} образцов");
            Console.WriteLine("");
            
            var optimizer = new ConstrainedOptimizerGraph(
                nct: ncts[targetNct],
                key: keys[targetNct],
                graphJsonPath: graphJsonPath,
                targetNct: targetNct,
                learningRate: learningRate,
                stepSize: stepSize,
                earlyStopping: earlyStopping
            );
            Console.WriteLine("");
            
            int batchEnd = batchSize > 0 ? Math.Min(batchSize, data.Count) : data.Count;
            
            var allMetrics = new List<AttackMetrics>();
            var adversarialImages = new List<double[]>();
            
            for (int idx = 0; idx < batchEnd; idx++)
            {
                //int id = data[idx].Item1;
                //int Class = data[idx].Item2;

                Console.WriteLine($"ОБРАЗЕЦ {idx + 1}/{batchEnd}:");
                Console.WriteLine($"  - Целевой класс: {targetNct}");
                Console.WriteLine($"  - Макс итераций: {nIterations}");


                double[] features = data[idx].Item3;
                var (adversarialImage, metrics) = optimizer.Attack(
                    image: features,
                    trueClass: targetNct,
                    nIterations: nIterations,
                    verbose: true
                );
                
                metrics.SampleIndex = idx;
                adversarialImages.Add(adversarialImage);
                allMetrics.Add(metrics);
                
                Console.WriteLine($"[DONE] Результат:");
                Console.WriteLine($"  - Исходное расстояние: {metrics.InitialHammingDistance}");
                Console.WriteLine($"  - Финальное расстояние: {metrics.FinalHammingDistance}");
                Console.WriteLine($"  - Улучшение: {metrics.Improvement}");
            }
            
            Directory.CreateDirectory(outputDir);
            
            string metricsPath = Path.Combine(outputDir, "metrics.json");
            var resultsJson = new
            {
                timestamp = DateTime.UtcNow.ToString("O"),
                target_nct = targetNct,
                attack_parameters = new
                {
                    learning_rate = learningRate,
                    step_size = stepSize,
                    n_iterations = nIterations
                },
                metrics = allMetrics
            };
            
            File.WriteAllText(metricsPath, JsonConvert.SerializeObject(resultsJson, Formatting.Indented));
            Console.WriteLine($"[DONE] Метрики: {metricsPath}");
            
            string adversarialPath = Path.Combine(outputDir, "adversarial_samples.json");
            var adversarialData = new
            {
                count = adversarialImages.Count,
                feature_count = adversarialImages.Count > 0 ? adversarialImages[0].Length : 0,
                samples = adversarialImages.Select((img, idx) => new
                {
                    index = idx,
                    features = img
                }).ToList()
            };
            
            File.WriteAllText(adversarialPath, JsonConvert.SerializeObject(adversarialData, Formatting.Indented));
            Console.WriteLine($"[DONE] Состязательные примеры: {adversarialPath}");

            Console.WriteLine("");
            Console.WriteLine("СТАТИСТИКА АТАКИ");
            
            var initialDistances = allMetrics.Select(m => m.InitialHammingDistance).ToList();
            var finalDistances = allMetrics.Select(m => m.FinalHammingDistance).ToList();
            var improvements = allMetrics.Select(m => m.Improvement).ToList();
            
            Console.WriteLine($"  - Атаковано образцов: {allMetrics.Count}");
            Console.WriteLine($"  - Среднее исходное расстояние: {initialDistances.Average():F2}");
            Console.WriteLine($"  - Среднее финальное расстояние: {finalDistances.Average():F2}");
            Console.WriteLine($"  - Среднее улучшение: {improvements.Average():F2}");
            
            int successCount = allMetrics.Count(m => m.Improvement > 0);
            double successRate = (double)successCount / allMetrics.Count;
            // Console.WriteLine($"  - Успешность: {successRate:P1}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("ATTACK FAILED");
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    private static List<(int id, int trueClass, double[] features)> LoadDataFromCsv(string csvPath, int featureCount)
    {
        var data = new List<(int, int, double[])>();
        
        using (var reader = new StreamReader(csvPath))
        {
            string headerLine = reader.ReadLine();
            
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split(',');
                
                if (parts.Length < featureCount + 3)
                    continue;
                
                int id = int.Parse(parts[0]);
                int trueClass = int.Parse(parts[1]);
                
                var features = new double[featureCount];
                for (int i = 0; i < featureCount; i++)
                {
                    features[i] = double.Parse(parts[i + 3], CultureInfo.InvariantCulture);
                }
                
                data.Add((id, trueClass, features));
            }
        }
        
        return data;
    }
    
    private static void PrintHelp()
    {
        Console.WriteLine("");
        Console.WriteLine("Usage: dotnet run -- attack [options]");
        Console.WriteLine("");
        Console.WriteLine("Required Options:");
        Console.WriteLine("  --graph-json <path>         Path to correlation graph JSON");
        Console.WriteLine("  --model <path>              Path to model.json");
        Console.WriteLine("  --input <path>              Path to data_processed.csv");
        Console.WriteLine("  --output <path>             Output directory");
        Console.WriteLine("");
        Console.WriteLine("Optional Options:");
        Console.WriteLine("  --learning-rate <double>    Learning rate (default: 0.01)");
        Console.WriteLine("  --step-size <double>        Step size (default: 1.0)");
        Console.WriteLine("  --n-iterations <int>        Number of iterations (default: 100)");
        Console.WriteLine("  --early-stopping <int>      Patience: iterations without improvement (default: 30)");
        Console.WriteLine("  --batch-size <int>          Batch size (default: 10, 0 = all)");
        Console.WriteLine("  --target-nct <int>          Target NCT index (default: 0)");
        Console.WriteLine("  --help                      Show this help message");
        Console.WriteLine("");
    }
}
