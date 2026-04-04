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
    
    private readonly double decayRate;
    private readonly double momentum;
    private readonly double maxL2;
    
    private readonly double importanceThreshold;
    private readonly List<string> parentFeatures;
    
    
    public ConstrainedOptimizerGraph(
        NCT nct,
        BitArray key,
        string graphJsonPath,
        int targetNct = 0,
        double learningRate = 0.01,
        double stepSize = 1.0,
        int earlyStopping = 30,
        bool earlyStop = false,
        double decayRate = 0.003,
        double momentum = 0.5,
        double maxL2 = 0.0
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
        this.decayRate = decayRate;
        this.momentum = momentum;
        this.maxL2 = maxL2;
        
        this.graph = LoadGraph(graphJsonPath);
        this.importanceThreshold = ComputeImportanceThreshold();
        this.parentFeatures = GetParentFeatures();

        Console.WriteLine($"[DONE] Граф загружен");
        Console.WriteLine($"  - Порог importance: {importanceThreshold:F4}");
        Console.WriteLine($"  - Признаков с importance >= порога: {parentFeatures.Count}");
        Console.WriteLine($"  - Decay rate: {decayRate}");
        Console.WriteLine($"  - Momentum: {momentum}");
        Console.WriteLine($"  - Max L2: {(maxL2 > 0 ? maxL2.ToString("F2") : "off")}");
    }
    
    private CorrelationGraph LoadGraph(string graphJsonPath)
    {
        string json = File.ReadAllText(graphJsonPath);
        return JsonConvert.DeserializeObject<CorrelationGraph>(json); 
    }
    
    private double ComputeImportanceThreshold()
    {
        var nonZeroImportance = graph.Features.Values
            .Select(f => f.Importance)
            .Where(imp => imp > 0.0)
            .ToList();
        
        double threshold = nonZeroImportance.Average();
        return threshold;
    }
    
    private List<string> GetParentFeatures()
    {
        var parents = graph.Features
            .Where(kvp => kvp.Value.Importance >= importanceThreshold)
            .OrderByDescending(kvp => kvp.Value.Importance)
            .Select(kvp => kvp.Key)
            .ToList();
        
        return parents;
    }
    
    public (double[] adversarialImage, AttackMetrics metrics) Attack(
        double[] image,
        int trueClass,
        int nIterations = 100,
        bool verbose = true
    )
    {
        double[] originalImage = (double[])image.Clone();
        double[] currentImage = (double[])image.Clone();
        double[] bestImage = (double[])image.Clone();
        var distancesHistory = new List<int>();
        int bestHamming = ComputeHammingDistance(bestImage, trueClass);
        int patienceCounter = 0;
        
        int featureCount = currentImage.Length;
        double[] velocity = new double[featureCount];
        
        for (int iteration = 0; iteration < nIterations; iteration++)
        {
            int previousBest = bestHamming;
            int currentDistance = ComputeHammingDistance(currentImage, trueClass);
            distancesHistory.Add(currentDistance);
            
            if (currentDistance < bestHamming)
            {
                bestHamming = currentDistance;
                Array.Copy(currentImage, bestImage, currentImage.Length);
            }
            
            if (verbose && (iteration % 100 == 0 || iteration == 0 || iteration == nIterations - 1))
            {
                int delta = distancesHistory[0] - bestHamming;
                double l2Dist = ComputeL2Distance(currentImage, originalImage);
                Console.WriteLine($"  Итерация {iteration,4:D}: Hamming = {currentDistance,3:D} (best: {bestHamming,3:D}, Δ: {delta,3:D}, L2: {l2Dist:F3})");
            }
            
            if (earlyStop)
            {
                if (currentDistance < previousBest)
                    patienceCounter = 0;
                else
                    patienceCounter++;
                
                if (patienceCounter >= earlyStopping)
                {
                    Console.WriteLine(
                        $"Early stopping: {earlyStopping} итераций без улучшения текущего состояния");
                    return (bestImage, BuildMetricsFromBest(
                        distancesHistory,
                        iteration,
                        bestHamming,
                        ComputeL2Distance(bestImage, originalImage),
                        earlyStoppedFlag: true,
                        reason: $"No strict improvement on current iterate for {earlyStopping} iterations"
                    ));
                }
            }
            
            double decay = 1.0 / (1.0 + decayRate * iteration);
            double adaptiveLR = learningRate * decay;
            
            foreach (string parentIdStr in parentFeatures)
            {
                GraphFeature parentFeat = graph.Features[parentIdStr];

                if (int.TryParse(parentIdStr, out int parentId) && parentId >= 0 && parentId < currentImage.Length)
                {
                    double parentImportance = parentFeat.Importance;
                    double pMaxChange = (1.0 - parentImportance) * adaptiveLR * stepSize * 0.3;
                    double pGradient = ComputeGradientCentral(parentId, trueClass, currentImage);
                    
                    double direction = pGradient > 0 ? -1.0 : 1.0;
                    velocity[parentId] = momentum * velocity[parentId] + direction * pMaxChange;
                    currentImage[parentId] += velocity[parentId];
                }

                Dictionary<string, double> partners = parentFeat.Partners;
                
                foreach (var kvp in partners)
                {
                    string partnerIdStr = kvp.Key;
                    double partnerImportance = kvp.Value;
                    
                    if (!int.TryParse(partnerIdStr, out int partnerId))
                        continue;
                    
                    if (partnerId < 0 || partnerId >= currentImage.Length)
                        continue;
                    
                    double maxChange = (1.0 - partnerImportance) * adaptiveLR * stepSize;
                    double gradient = ComputeGradientCentral(partnerId, trueClass, currentImage);
                    
                    double dir = gradient > 0 ? -1.0 : 1.0;
                    velocity[partnerId] = momentum * velocity[partnerId] + dir * maxChange;
                    currentImage[partnerId] += velocity[partnerId];
                }
            }
            
            if (maxL2 > 0)
            {
                ProjectOntoL2Ball(currentImage, originalImage, maxL2);
            }
        }
        
        int finalH = ComputeHammingDistance(bestImage, trueClass);
        double finalL2 = ComputeL2Distance(bestImage, originalImage);
        return (bestImage, BuildMetricsFromBest(
            distancesHistory,
            nIterations,
            finalH,
            finalL2,
            earlyStoppedFlag: false,
            reason: "Max iterations reached"
        ));
    }
    
    private void ProjectOntoL2Ball(double[] currentImage, double[] originalImage, double radius)
    {
        double l2 = ComputeL2Distance(currentImage, originalImage);
        if (l2 > radius)
        {
            double scale = radius / l2;
            for (int i = 0; i < currentImage.Length; i++)
                currentImage[i] = originalImage[i] + (currentImage[i] - originalImage[i]) * scale;
        }
    }
    
    private double ComputeL2Distance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
    
    private int ComputeHammingDistance(double[] image, int trueClass)
    {
        BitArray code = nct.VerifyImage(image);
        BitArray keyBits = key;
        
        int distance = 0;
        int minLength = Math.Min(code.Count, keyBits.Count);
        for (int i = 0; i < minLength; i++)
            if (code[i] != keyBits[i])
                distance++;
        return distance;
    }
    
    private double ComputeGradientCentral(int featureId, int trueClass, double[] image)
    {
        double epsilon = 0.01;
        
        double[] imagePlus = (double[])image.Clone();
        imagePlus[featureId] += epsilon;
        int distancePlus = ComputeHammingDistance(imagePlus, trueClass);
        
        double[] imageMinus = (double[])image.Clone();
        imageMinus[featureId] -= epsilon;
        int distanceMinus = ComputeHammingDistance(imageMinus, trueClass);
        
        double gradient = (distancePlus - distanceMinus) / (2.0 * epsilon);
        return gradient;
    }
    
    private AttackMetrics BuildMetricsFromBest(
        List<int> distancesHistory,
        int iterationsCompleted,
        int bestHammingDistance,
        double l2Distance,
        bool earlyStoppedFlag,
        string reason
    )
    {
        return new AttackMetrics
        {
            InitialHammingDistance = distancesHistory[0],
            FinalHammingDistance = bestHammingDistance,
            Improvement = distancesHistory[0] - bestHammingDistance,
            IterationsCompleted = iterationsCompleted,
            DistancesHistory = distancesHistory,
            StoppedEarly = earlyStoppedFlag,
            Reason = reason,
            LearningRate = learningRate,
            StepSize = stepSize,
            L2Distance = l2Distance,
            DecayRate = decayRate,
            Momentum = momentum,
            MaxL2 = maxL2
        };
    }
}


// =====================================================================
//  WHITE-BOX НЕЙРОН-ТАРГЕТИРОВАННАЯ АТАКА
//  Вместо слепого градиента по дискретному Хэммингу,
//  вычисляем градиент непрерывного выхода каждого нейрона
//  и двигаем его в целевой интервал пороговой активации.
// =====================================================================

public class NeuronTargetedOptimizer
{
    // NCT-объект (для верификации через VerifyImage)
    private readonly NCT nct;
    private readonly BitArray key;
    
    // Внутренности НКП, загруженные из meta.json
    private int[][] synJ;       // synJ[neuron][pair] = индекс j
    private int[][] synT;       // synT[neuron][pair] = индекс t
    private double[][] weights; // weights[neuron][pair]
    private double[][] thresholds; // thresholds[neuron][3]
    private int[] tableIndices; // tableIndices[neuron]
    private double[] sxStranger; // sxStranger[feature]
    private bool[] keyBits;     // keyBits[128]
    private int neuronCount;    // 64
    private int featureCount;   // 512
    private int inputsPerNeuron; // 4
    
    // Параметры атаки
    private readonly double learningRate;
    private readonly double stepSize;
    private readonly double decayRate;
    private readonly double momentum;
    private readonly double maxL2;
    private readonly double protectionWeight; // вес защиты правильных нейронов
    
    private const double P = 0.9; // параметр нормализации
    
    // ========== ТАБЛИЦЫ ПРЕОБРАЗОВАНИЙ (из NCT_original.cs) ==========
    private static readonly bool[][][] TablesPatterns = new bool[][][] {
        new bool[][] { new bool[] { true, true }, new bool[] { false, false }, new bool[] { true, false }, new bool[] { false, true } },
        new bool[][] { new bool[] { true, true }, new bool[] { true, false }, new bool[] { false, false }, new bool[] { false, true } },
        new bool[][] { new bool[] { false, false }, new bool[] { true, true }, new bool[] { true, false }, new bool[] { false, true } },
        new bool[][] { new bool[] { false, false }, new bool[] { true, false }, new bool[] { true, true }, new bool[] { false, true } },
        new bool[][] { new bool[] { true, false }, new bool[] { false, false }, new bool[] { true, true }, new bool[] { false, true } },
        new bool[][] { new bool[] { true, false }, new bool[] { true, true }, new bool[] { false, false }, new bool[] { false, true } },
        new bool[][] { new bool[] { false, true }, new bool[] { false, false }, new bool[] { true, false }, new bool[] { true, true } },
        new bool[][] { new bool[] { false, true }, new bool[] { true, false }, new bool[] { false, false }, new bool[] { true, true } },
        new bool[][] { new bool[] { false, false }, new bool[] { false, true }, new bool[] { true, false }, new bool[] { true, true } },
        new bool[][] { new bool[] { false, false }, new bool[] { true, false }, new bool[] { false, true }, new bool[] { true, true } },
        new bool[][] { new bool[] { true, false }, new bool[] { false, false }, new bool[] { false, true }, new bool[] { true, true } },
        new bool[][] { new bool[] { true, false }, new bool[] { false, true }, new bool[] { false, false }, new bool[] { true, true } },
        new bool[][] { new bool[] { true, true }, new bool[] { false, true }, new bool[] { true, false }, new bool[] { false, false } },
        new bool[][] { new bool[] { true, true }, new bool[] { true, false }, new bool[] { false, true }, new bool[] { false, false } },
        new bool[][] { new bool[] { false, true }, new bool[] { true, true }, new bool[] { true, false }, new bool[] { false, false } },
        new bool[][] { new bool[] { false, true }, new bool[] { true, false }, new bool[] { true, true }, new bool[] { false, false } },
        new bool[][] { new bool[] { true, false }, new bool[] { false, true }, new bool[] { true, true }, new bool[] { false, false } },
        new bool[][] { new bool[] { true, false }, new bool[] { true, true }, new bool[] { false, true }, new bool[] { false, false } },
        new bool[][] { new bool[] { true, true }, new bool[] { false, false }, new bool[] { false, true }, new bool[] { true, false } },
        new bool[][] { new bool[] { true, true }, new bool[] { false, true }, new bool[] { false, false }, new bool[] { true, false } },
        new bool[][] { new bool[] { false, false }, new bool[] { true, true }, new bool[] { false, true }, new bool[] { true, false } },
        new bool[][] { new bool[] { false, false }, new bool[] { false, true }, new bool[] { true, true }, new bool[] { true, false } },
        new bool[][] { new bool[] { false, true }, new bool[] { false, false }, new bool[] { true, true }, new bool[] { true, false } },
        new bool[][] { new bool[] { false, true }, new bool[] { true, true }, new bool[] { false, false }, new bool[] { true, false } }
    };
    
    public NeuronTargetedOptimizer(
        string modelJsonPath,
        int targetNctIdx,
        NCT nct,
        BitArray key,
        double learningRate = 0.01,
        double stepSize = 1.0,
        double decayRate = 0.001,
        double momentum = 0.5,
        double maxL2 = 0.0,
        double protectionWeight = 0.3
    )
    {
        this.nct = nct;
        this.key = key;
        this.learningRate = learningRate;
        this.stepSize = stepSize;
        this.decayRate = decayRate;
        this.momentum = momentum;
        this.maxL2 = maxL2;
        this.protectionWeight = protectionWeight;
        
        // Загрузить внутренности НКП из meta.json
        LoadNctInternals(modelJsonPath, targetNctIdx);
        
        Console.WriteLine($"[WHITEBOX] НКП загружен из meta.json:");
        Console.WriteLine($"  - Нейронов: {neuronCount}");
        Console.WriteLine($"  - Признаков: {featureCount}");
        Console.WriteLine($"  - Входов на нейрон: {inputsPerNeuron}");
        Console.WriteLine($"  - Decay rate: {decayRate}");
        Console.WriteLine($"  - Momentum: {momentum}");
        Console.WriteLine($"  - Max L2: {(maxL2 > 0 ? maxL2.ToString("F2") : "off")}");
    }
    
    private void LoadNctInternals(string modelJsonPath, int targetNctIdx)
    {
        string json = File.ReadAllText(modelJsonPath);
        dynamic model = JsonConvert.DeserializeObject(json);
        dynamic nctJson = model.ncts[targetNctIdx];
        
        // Feature count и neuron count
        featureCount = Convert.ToInt32(model.feature_count);
        
        // Weights
        var weightsArr = JsonConvert.DeserializeObject<double[][]>(nctJson.weights.ToString());
        weights = weightsArr;
        neuronCount = weights.Length;
        inputsPerNeuron = weights[0].Length;
        
        // Thresholds  
        thresholds = JsonConvert.DeserializeObject<double[][]>(nctJson.thresholds.ToString());
        
        // Table indices
        tableIndices = JsonConvert.DeserializeObject<int[]>(nctJson.table_indices.ToString());
        
        // sx_stranger
        sxStranger = JsonConvert.DeserializeObject<double[]>(nctJson.sx_stranger.ToString());
        
        // Synapses → разделяем на synJ и synT для быстрого доступа
        var synapsesData = JsonConvert.DeserializeObject<int[][][]>(nctJson.synapses.ToString());
        synJ = new int[neuronCount][];
        synT = new int[neuronCount][];
        for (int n = 0; n < neuronCount; n++)
        {
            synJ[n] = new int[inputsPerNeuron];
            synT[n] = new int[inputsPerNeuron];
            for (int k = 0; k < inputsPerNeuron; k++)
            {
                synJ[n][k] = synapsesData[n][k][0];
                synT[n][k] = synapsesData[n][k][1];
            }
        }
        
        // Key bits
        string keyStr = nctJson.key_bits.ToString();
        keyBits = new bool[keyStr.Length];
        for (int i = 0; i < keyStr.Length; i++)
            keyBits[i] = keyStr[i] == '1';
    }
    
    // ==================== РЕПЛИКА ВЫЧИСЛЕНИЙ NCT ====================
    
    /// <summary>
    /// Нормализация: norm[j] = (|x[j]| / sx[j])^p
    /// </summary>
    private double[] NormalizeFeatures(double[] rawFeatures)
    {
        double[] norm = new double[rawFeatures.Length];
        for (int j = 0; j < rawFeatures.Length; j++)
        {
            if (sxStranger[j] != 0)
                norm[j] = Math.Pow(Math.Abs(rawFeatures[j]) / sxStranger[j], P);
            else
                norm[j] = 0;
        }
        return norm;
    }
    
    /// <summary>
    /// Вычислить выход нейрона — точная реплика NCT.GetNeuronOutput
    /// </summary>
    private double ComputeNeuronOutput(int neuronIdx, double[] normFeatures)
    {
        int nInputs = inputsPerNeuron;
        double[] w = weights[neuronIdx];
        
        // 1. Мета-признаки первого порядка: ||norm[j]| - |norm[t]||
        double[] meta = new double[nInputs];
        double my = Math.Abs(Math.Abs(normFeatures[synJ[neuronIdx][0]]) - Math.Abs(normFeatures[synT[neuronIdx][0]]));
        meta[0] = my;
        for (int i = 1; i < nInputs; i++)
        {
            meta[i] = Math.Abs(Math.Abs(normFeatures[synJ[neuronIdx][i]]) - Math.Abs(normFeatures[synT[neuronIdx][i]]));
            // Рекурентное среднее: my = ((i)/(i+1))*my + (1/(i+1))*meta[i]
            my = ((double)i / (i + 1)) * my + (1.0 / (i + 1)) * meta[i];
        }
        
        // 2. Мета-признаки второго порядка: (meta[k] - my)^2 * w[k], потом среднее, потом sqrt
        double my3 = Math.Pow(meta[0] - my, 2) * w[0];
        for (int i = 1; i < nInputs; i++)
        {
            double val = Math.Pow(meta[i] - my, 2) * w[i];
            my3 = ((double)i / (i + 1)) * my3 + (1.0 / (i + 1)) * val;
        }
        
        return Math.Sqrt(Math.Abs(my3)); // Abs для защиты от -0
    }
    
    /// <summary>
    /// Определить текущий интервал по порогам (0..3)
    /// </summary>
    private int GetInterval(double y, int neuronIdx)
    {
        double[] th = thresholds[neuronIdx];
        if (y < th[0]) return 0;
        if (y < th[1]) return 1;
        if (y < th[2]) return 2;
        return 3;
    }
    
    /// <summary>
    /// Найти ближайший целевой интервал, который выдаёт правильные биты ключа
    /// </summary>
    private int GetClosestTargetInterval(int neuronIdx, double currentY)
    {
        bool b0 = keyBits[2 * neuronIdx];
        bool b1 = keyBits[2 * neuronIdx + 1];
        int tableIdx = tableIndices[neuronIdx];
        
        int bestInterval = -1;
        double bestDistance = double.MaxValue;
        
        for (int interval = 0; interval < 4; interval++)
        {
            if (TablesPatterns[tableIdx][interval][0] == b0 &&
                TablesPatterns[tableIdx][interval][1] == b1)
            {
                double center = GetIntervalCenter(thresholds[neuronIdx], interval);
                double dist = Math.Abs(currentY - center);
                if (dist < bestDistance)
                {
                    bestDistance = dist;
                    bestInterval = interval;
                }
            }
        }
        return bestInterval;
    }
    
    /// <summary>
    /// Целевое значение y для нейрона: ближайшая граница целевого интервала + маленький отступ.
    /// Минимизирует необходимое движение и снижает коллизии с другими нейронами.
    /// </summary>
    private double GetTargetY(double[] th, int targetInterval, double currentY)
    {
        double margin = 0.005; // маленький отступ за порог
        double lo, hi;
        switch (targetInterval)
        {
            case 0: lo = 0; hi = th[0]; break;
            case 1: lo = th[0]; hi = th[1]; break;
            case 2: lo = th[1]; hi = th[2]; break;
            default: lo = th[2]; hi = th[2] + (th[2] - th[1]); break;
        }
        // Если текущий y выше интервала → целимся в верхнюю границу минус margin
        // Если ниже → в нижнюю + margin
        if (currentY >= hi)
            return hi - margin;
        else if (currentY < lo)
            return lo + margin;
        else
            return (lo + hi) / 2.0; // уже внутри (маловероятно, т.к. это «неправильный» нейрон)
    }
    
    /// <summary>
    /// Расстояние от y до ближайшей границы интервала (для защиты правильных нейронов)
    /// </summary>
    private double DistanceToNearestBoundary(double y, double[] th, int interval)
    {
        double distLow, distHigh;
        switch (interval)
        {
            case 0: return th[0] - y; // расстояние до верхней границы
            case 1: distLow = y - th[0]; distHigh = th[1] - y; return Math.Min(distLow, distHigh);
            case 2: distLow = y - th[1]; distHigh = th[2] - y; return Math.Min(distLow, distHigh);
            default: return y - th[2]; // расстояние до нижней границы
        }
    }
    
    /// <summary>
    /// Центр интервала (для защиты правильных нейронов)
    /// </summary>
    private double GetIntervalCenter(double[] th, int interval)
    {
        switch (interval)
        {
            case 0: return th[0] * 0.5;
            case 1: return (th[0] + th[1]) / 2.0;
            case 2: return (th[1] + th[2]) / 2.0;
            case 3: return th[2] + (th[2] - th[1]) * 0.5;
            default: return 0;
        }
    }
    
    /// <summary>
    /// Эффективный градиент выхода нейрона по входному признаку.
    /// Перенормируем только изменённый признак, не весь вектор.
    /// </summary>
    private double ComputeNeuronGradient(int neuronIdx, int featureIdx, double[] rawFeatures, double[] normFeatures)
    {
        double epsilon = 0.001;
        
        // Сохраняем оригинальное нормализованное значение
        double origNorm = normFeatures[featureIdx];
        double sx = sxStranger[featureIdx];
        if (sx == 0) return 0;
        
        // Пертурбация +ε
        double rawPlus = rawFeatures[featureIdx] + epsilon;
        normFeatures[featureIdx] = Math.Pow(Math.Abs(rawPlus) / sx, P);
        double yPlus = ComputeNeuronOutput(neuronIdx, normFeatures);
        
        // Пертурбация -ε
        double rawMinus = rawFeatures[featureIdx] - epsilon;
        normFeatures[featureIdx] = Math.Pow(Math.Abs(rawMinus) / sx, P);
        double yMinus = ComputeNeuronOutput(neuronIdx, normFeatures);
        
        // Восстановление
        normFeatures[featureIdx] = origNorm;
        
        return (yPlus - yMinus) / (2.0 * epsilon);
    }
    
    /// <summary>
    /// Вычислить расстояние Хэмминга через NCT.VerifyImage (для верификации)
    /// </summary>
    private int ComputeHammingViaNCT(double[] image)
    {
        BitArray code = nct.VerifyImage(image);
        int distance = 0;
        int minLength = Math.Min(code.Count, key.Count);
        for (int i = 0; i < minLength; i++)
            if (code[i] != key[i])
                distance++;
        return distance;
    }
    
    /// <summary>
    /// Посчитать число неправильных нейронов (внутренний расчёт)
    /// </summary>
    private int CountWrongNeurons(double[] normFeatures)
    {
        int wrong = 0;
        for (int n = 0; n < neuronCount; n++)
        {
            double y = ComputeNeuronOutput(n, normFeatures);
            int currentInterval = GetInterval(y, n);
            int targetInterval = GetClosestTargetInterval(n, y);
            if (currentInterval != targetInterval)
                wrong++;
        }
        return wrong;
    }
    
    // ==================== ОСНОВНОЙ МЕТОД АТАКИ ====================
    
    public (double[] adversarialImage, AttackMetrics metrics) Attack(
        double[] image,
        int trueClass,
        int nIterations = 100,
        bool verbose = true
    )
    {
        double[] originalImage = (double[])image.Clone();
        double[] currentImage = (double[])image.Clone();
        double[] bestImage = (double[])image.Clone();
        int bestHamming = ComputeHammingViaNCT(bestImage);
        var distancesHistory = new List<int>();
        double[] velocity = new double[featureCount];
        
        for (int iteration = 0; iteration < nIterations; iteration++)
        {
            int currentHamming = ComputeHammingViaNCT(currentImage);
            distancesHistory.Add(currentHamming);
            
            if (currentHamming < bestHamming)
            {
                bestHamming = currentHamming;
                Array.Copy(currentImage, bestImage, currentImage.Length);
            }
            
            if (verbose && (iteration % 50 == 0 || iteration == 0 || iteration == nIterations - 1))
            {
                double l2 = ComputeL2(currentImage, originalImage);
                double[] normForLog = NormalizeFeatures(currentImage);
                int wrongN = CountWrongNeurons(normForLog);
                Console.WriteLine($"  Итерация {iteration,4}: HD = {currentHamming,3} (best: {bestHamming,3}, wrong neurons: {wrongN}/{neuronCount}, L2: {l2:F3})");
            }
            
            // Adaptive learning rate
            double decay = 1.0 / (1.0 + decayRate * iteration);
            double lr = learningRate * decay;
            
            // Нормализуем один раз для всех нейронов
            double[] normFeatures = NormalizeFeatures(currentImage);
            
            // Аккумулируем дельты по всем неправильным нейронам
            double[] delta = new double[featureCount];
            int wrongCount = 0;
            
            for (int n = 0; n < neuronCount; n++)
            {
                double y = ComputeNeuronOutput(n, normFeatures);
                int currentInterval = GetInterval(y, n);
                int targetInterval = GetClosestTargetInterval(n, y);
                
                if (currentInterval == targetInterval)
                {
                    // ===== ЗАЩИТА ПРАВИЛЬНЫХ НЕЙРОНОВ =====
                    // Если нейрон близко к границе, добавляем защитный градиент
                    double distBoundary = DistanceToNearestBoundary(y, thresholds[n], currentInterval);
                    double intervalWidth = (currentInterval == 0) ? thresholds[n][0] :
                                           (currentInterval == 3) ? (thresholds[n][2] - thresholds[n][1]) :
                                           (thresholds[n][currentInterval] - thresholds[n][currentInterval - 1]);
                    
                    // Если меньше 30% ширины интервала от границы → защищаем
                    if (intervalWidth > 0 && distBoundary < intervalWidth * 0.3)
                    {
                        double centerY = GetIntervalCenter(thresholds[n], currentInterval);
                        double protError = (centerY - y) * protectionWeight;
                        
                        var involved = new HashSet<int>();
                        for (int k = 0; k < inputsPerNeuron; k++)
                        {
                            involved.Add(synJ[n][k]);
                            involved.Add(synT[n][k]);
                        }
                        foreach (int f in involved)
                        {
                            if (f >= 0 && f < featureCount)
                            {
                                double grad = ComputeNeuronGradient(n, f, currentImage, normFeatures);
                                delta[f] += lr * protError * grad * stepSize;
                            }
                        }
                    }
                    continue;
                }
                
                wrongCount++;
                
                // ===== БЛИЖАЙШАЯ ГРАНИЦА (вместо центра) =====
                double targetY = GetTargetY(thresholds[n], targetInterval, y);
                double error = targetY - y; // знаковая ошибка
                
                // Для каждого признака, задействованного в этом нейроне
                var involvedWrong = new HashSet<int>();
                for (int k = 0; k < inputsPerNeuron; k++)
                {
                    involvedWrong.Add(synJ[n][k]);
                    involvedWrong.Add(synT[n][k]);
                }
                
                foreach (int f in involvedWrong)
                {
                    if (f >= 0 && f < featureCount)
                    {
                        double grad = ComputeNeuronGradient(n, f, currentImage, normFeatures);
                        delta[f] += lr * error * grad * stepSize;
                    }
                }
            }
            
            // Применяем дельты с momentum
            for (int f = 0; f < featureCount; f++)
            {
                velocity[f] = momentum * velocity[f] + delta[f];
                currentImage[f] += velocity[f];
            }
            
            // L2-проекция
            if (maxL2 > 0)
                ProjectOntoL2Ball(currentImage, originalImage, maxL2);
        }
        
        int finalH = ComputeHammingViaNCT(bestImage);
        double finalL2 = ComputeL2(bestImage, originalImage);
        return (bestImage, new AttackMetrics
        {
            InitialHammingDistance = distancesHistory[0],
            FinalHammingDistance = finalH,
            Improvement = distancesHistory[0] - finalH,
            IterationsCompleted = nIterations,
            DistancesHistory = distancesHistory,
            StoppedEarly = false,
            Reason = "Max iterations reached",
            LearningRate = learningRate,
            StepSize = stepSize,
            L2Distance = finalL2,
            DecayRate = decayRate,
            Momentum = momentum,
            MaxL2 = maxL2
        });
    }
    
    private double ComputeL2(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
    
    private void ProjectOntoL2Ball(double[] current, double[] original, double radius)
    {
        double l2 = ComputeL2(current, original);
        if (l2 > radius)
        {
            double scale = radius / l2;
            for (int i = 0; i < current.Length; i++)
                current[i] = original[i] + (current[i] - original[i]) * scale;
        }
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
    
    [JsonProperty("l2_distance")]
    public double L2Distance { get; set; }
    
    [JsonProperty("decay_rate")]
    public double DecayRate { get; set; }
    
    [JsonProperty("momentum")]
    public double Momentum { get; set; }
    
    [JsonProperty("max_l2")]
    public double MaxL2 { get; set; }
    
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
        string fullDataCsv = null;
        int targetPhoto = 11;
        int nPerClass = 14;
        int nTrain = 10;
        
        double decayRate = 0.001;
        double momentumParam = 0.5;
        double maxL2 = 0.0;
        
        // Режим атаки: "graph" (black-box) или "whitebox" (нейрон-таргетированная)
        string mode = "whitebox";
        
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
            else if (args[i] == "--full-data" && i + 1 < args.Length)
                fullDataCsv = args[++i];
            else if (args[i] == "--target-photo" && i + 1 < args.Length)
                targetPhoto = int.Parse(args[++i]);
            else if (args[i] == "--decay-rate" && i + 1 < args.Length)
                decayRate = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--momentum" && i + 1 < args.Length)
                momentumParam = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--max-l2" && i + 1 < args.Length)
                maxL2 = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--mode" && i + 1 < args.Length)
                mode = args[++i];
            else if (args[i] == "--help")
            {
                PrintHelp();
                return;
            }
        }
        
        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(inputCsv) || string.IsNullOrEmpty(outputDir))
        {
            Console.WriteLine("Error: Missing required arguments");
            PrintHelp();
            return;
        }
        
        // Для graph-mode нужен граф
        if (mode == "graph" && string.IsNullOrEmpty(graphJsonPath))
        {
            Console.WriteLine("Error: --graph-json required for graph mode");
            PrintHelp();
            return;
        }
        
        try
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model not found: {modelPath}");
            if (!File.Exists(inputCsv))
                throw new FileNotFoundException($"Input CSV not found: {inputCsv}");
            
            
            var (ncts, keys, meta) = NctCliProgram.LoadModelFromJson(modelPath);
            Console.WriteLine($"[DONE] Модель загружена: {modelPath}");
            Console.WriteLine($"  - NCTs: {ncts.Length}");
            Console.WriteLine($"  - Feature count: {meta.feature_count}");
            Console.WriteLine($"  - Neurons: {meta.neurons_count}");
            Console.WriteLine($"  - Mode: {mode}");
            Console.WriteLine("");
            
            int featureCount = Convert.ToInt32(meta.feature_count);
            var data = LoadDataFromCsv(inputCsv, featureCount);
            Console.WriteLine($"[DONE] Данные из CSV загружены: {data.Count} образцов");
            Console.WriteLine("");
            
            // Создаём оптимизатор в зависимости от режима
            ConstrainedOptimizerGraph graphOptimizer = null;
            NeuronTargetedOptimizer whiteboxOptimizer = null;
            
            bool isGraphMode = (mode == "graph");
            if (isGraphMode)
            {
                graphOptimizer = new ConstrainedOptimizerGraph(
                    nct: ncts[targetNct],
                    key: keys[targetNct],
                    graphJsonPath: graphJsonPath,
                    targetNct: targetNct,
                    learningRate: learningRate,
                    stepSize: stepSize,
                    earlyStopping: earlyStopping,
                    decayRate: decayRate,
                    momentum: momentumParam,
                    maxL2: maxL2
                );
            }
            else // whitebox
            {
                whiteboxOptimizer = new NeuronTargetedOptimizer(
                    modelJsonPath: modelPath,
                    targetNctIdx: targetNct,
                    nct: ncts[targetNct],
                    key: keys[targetNct],
                    learningRate: learningRate,
                    stepSize: stepSize,
                    decayRate: decayRate,
                    momentum: momentumParam,
                    maxL2: maxL2
                );
            }
            Console.WriteLine("");

            double[] targetPhotoFeatures = null;
            BitArray targetPhotoCode = null;

            if (!string.IsNullOrEmpty(fullDataCsv) && File.Exists(fullDataCsv))
            {
                var fullData = LoadDataFromCsv(fullDataCsv, featureCount);
                var targetTestSamples = fullData.Where(x => x.trueClass == targetNct && (x.id % nPerClass) >= nTrain).ToList();
                
                if (targetTestSamples.Count > 0)
                {
                    double sumHd = 0;
                    foreach(var sample in targetTestSamples)
                    {
                        var hd = ComputeHammingDistance(ncts[targetNct].VerifyImage(sample.features), keys[targetNct]);
                        sumHd += hd;
                    }
                    Console.WriteLine($"[*] Среднее расстояние Хэмминга для целевого класса {targetNct} (split=test): {sumHd / targetTestSamples.Count:F2}");
                }

                int targetPhotoId = targetNct * nPerClass + (targetPhoto - 1);
                var tPhotoSample = fullData.FirstOrDefault(x => x.id == targetPhotoId);
                if (tPhotoSample.features != null)
                {
                    targetPhotoFeatures = tPhotoSample.features;
                    targetPhotoCode = ncts[targetNct].VerifyImage(targetPhotoFeatures);
                    int hdKey = ComputeHammingDistance(targetPhotoCode, keys[targetNct]);
                    Console.WriteLine($"[*] Расстояние Хэмминга для целевой фотографии {targetPhoto} (до ключа): {hdKey}");
                }
                Console.WriteLine("");
            }
            
            int batchEnd = batchSize > 0 ? Math.Min(batchSize, data.Count) : data.Count;
            Console.WriteLine($"  - Целевой класс: {targetNct}");
            Console.WriteLine($"  - Макс итераций: {nIterations}");
            Console.WriteLine("");
            
            var allMetrics = new List<AttackMetrics>();
            var adversarialImages = new List<double[]>();
            var adversarialRecords = new List<object>();
            
            for (int idx = 0; idx < batchEnd; idx++)
            {
                int id = data[idx].Item1;
                int trueClass = data[idx].Item2;

                int person = (id / nPerClass) + 1;
                int photo = (id % nPerClass) + 1;
                Console.WriteLine($"[{idx + 1}/{batchEnd}] Класс {person}, фото № {photo} :");

                double[] features = data[idx].Item3;
                
                (double[] adversarialImage, AttackMetrics metrics) result;
                
                if (isGraphMode)
                {
                    result = graphOptimizer.Attack(
                        image: features,
                        trueClass: targetNct,
                        nIterations: nIterations,
                        verbose: true
                    );
                }
                else // whitebox
                {
                    result = whiteboxOptimizer.Attack(
                        image: features,
                        trueClass: targetNct,
                        nIterations: nIterations,
                        verbose: true
                    );
                }
                
                var adversarialImage = result.adversarialImage;
                var metrics = result.metrics;
                
                metrics.SampleIndex = idx;
                adversarialImages.Add(adversarialImage);
                adversarialRecords.Add(new
                {
                    id = id,
                    true_class = trueClass,
                    features = adversarialImage
                });
                allMetrics.Add(metrics);
                
                Console.WriteLine($"[DONE] Результат:");
                Console.WriteLine($"  - Исходное расстояние: {metrics.InitialHammingDistance}");
                Console.WriteLine($"  - Финальное расстояние: {metrics.FinalHammingDistance}");
                Console.WriteLine($"  - Улучшение: {metrics.Improvement}");
                Console.WriteLine($"  - L2-расстояние: {metrics.L2Distance:F4}");

                if (targetPhotoCode != null)
                {
                    var finalCode = ncts[targetNct].VerifyImage(adversarialImage);
                    int hdToTarget = ComputeHammingDistance(finalCode, targetPhotoCode);
                    Console.WriteLine($"  - Расстояние между финальным откликом 'Чужого' и откликом целевой фотографии 'Своего': {hdToTarget}");
                }
            }
            
            Directory.CreateDirectory(outputDir);
            
            string metricsPath = Path.Combine(outputDir, "metrics.json");
            var resultsJson = new
            {
                timestamp = DateTime.UtcNow.ToString("O"),
                target_nct = targetNct,
                mode = mode,
                attack_parameters = new
                {
                    learning_rate = learningRate,
                    step_size = stepSize,
                    n_iterations = nIterations,
                    early_stopping_patience = earlyStopping,
                    decay_rate = decayRate,
                    momentum = momentumParam,
                    max_l2 = maxL2
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
                samples = adversarialRecords
            };
            
            File.WriteAllText(adversarialPath, JsonConvert.SerializeObject(adversarialData, Formatting.Indented));
            Console.WriteLine($"[DONE] Состязательные примеры: {adversarialPath}");

            Console.WriteLine("");
            Console.WriteLine("СТАТИСТИКА АТАКИ");
            
            var initialDistances = allMetrics.Select(m => m.InitialHammingDistance).ToList();
            var finalDistances = allMetrics.Select(m => m.FinalHammingDistance).ToList();
            var improvements = allMetrics.Select(m => m.Improvement).ToList();
            var l2Distances = allMetrics.Select(m => m.L2Distance).ToList();
            
            Console.WriteLine($"  - Атаковано образцов: {allMetrics.Count}");
            Console.WriteLine($"  - Среднее исходное расстояние: {initialDistances.Average():F2}");
            Console.WriteLine($"  - Среднее финальное расстояние: {finalDistances.Average():F2}");
            Console.WriteLine($"  - Среднее улучшение: {improvements.Average():F2}");
            Console.WriteLine($"  - Средняя L2-пертурбация: {l2Distances.Average():F4}");
            
            int successCount = allMetrics.Count(m => m.Improvement > 0);
            double successRate = (double)successCount / allMetrics.Count;
            Console.WriteLine($"  - Успешность: {successRate:P1}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("ATTACK FAILED");
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    private static int ComputeHammingDistance(BitArray code1, BitArray code2)
    {
        int distance = 0;
        int minLength = Math.Min(code1.Count, code2.Count);
        for (int i = 0; i < minLength; i++)
        {
            if (code1[i] != code2[i])
                distance++;
        }
        return distance;
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
        Console.WriteLine("Usage: dotnet run -- [options]");
        Console.WriteLine("");
        Console.WriteLine("Required Options:");
        Console.WriteLine("  --model <path>              Path to model (meta.json)");
        Console.WriteLine("  --input <path>              Path to data_processed.csv");
        Console.WriteLine("  --output <path>             Output directory");
        Console.WriteLine("");
        Console.WriteLine("Mode:");
        Console.WriteLine("  --mode <whitebox|graph>     Attack mode (default: whitebox)");
        Console.WriteLine("  --graph-json <path>         Path to graph.json (required for graph mode)");
        Console.WriteLine("");
        Console.WriteLine("Optional Options:");
        Console.WriteLine("  --learning-rate <double>    Learning rate (default: 0.01)");
        Console.WriteLine("  --step-size <double>        Step size (default: 1.0)");
        Console.WriteLine("  --n-iterations <int>        Number of iterations (default: 100)");
        Console.WriteLine("  --early-stopping <int>      Patience (default: 30, graph mode only)");
        Console.WriteLine("  --batch-size <int>          Batch size (default: 10, 0 = all)");
        Console.WriteLine("  --target-nct <int>          Target NCT index (default: 0)");
        Console.WriteLine("  --decay-rate <double>       LR decay rate (default: 0.003)");
        Console.WriteLine("  --momentum <double>         Momentum coefficient (default: 0.5)");
        Console.WriteLine("  --max-l2 <double>           Max L2 perturbation budget (default: 0 = off)");
        Console.WriteLine("  --help                      Show this help message");
        Console.WriteLine("");
    }
}
