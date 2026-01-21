using NCT_framework;
using System.Collections;
using Newtonsoft.Json;
using System.IO;
using System.Linq;
using System.Globalization;

class Program{        
        static void Main(string[] args){
            Random r = new Random();
            // Сгенерировали набор данных Свои (100 классов)
            List<List<double[]>> dataOwns=DataFactory.GetImgClassesWithCorralatedFeatures(r, 200, 10, 259, 0, 14);
            BitArray[] keys = new BitArray[10];
            NCT_framework.NCT[] NCTs = new NCT_framework.NCT[10];

        // Синтезируем НКП для каждого Своего
        for (int i = 0; i < 10; i++)
        {
            // Обучающая выборка Свой из 10 примеров
            List<double[]> owns = new List<double[]>();
            for (int k = 0; k < 10; k++)
            {
                int imgIndex = r.Next(0, dataOwns[i].Count);
                owns.Add(dataOwns[i][imgIndex]);
                dataOwns[i].RemoveAt(imgIndex);
            }
            // Обучающая выборка Чужой
            List<double[]> strangers = new List<double[]>();
            for (int k = 0; k < 10; k++)
                if (i != k)
                {
                    int index = r.Next(0, dataOwns[k].Count);
                    strangers.Add(dataOwns[k][index]);
                    dataOwns[k].RemoveAt(index);
                }
            // Генерируем ключ
            keys[i] = KeyFactory.GetKey(r, 128);

            // Создаем и обучаем НКП
            NCTs[i] = new NCT_framework.NCT();
            NCTs[i].Training(owns, strangers, 200, keys[i], 128, 4, r);
            // BitArray code = NCTs[i].VerifyImage(owns[0]);
            //     int hem = 0;
            //     for (int h = 0; h < code.Length; h++)
            //         if (keys[i][h] != code[h])
            //             hem++;
            //     Console.WriteLine($"Hem = {} for own[0] class={i}");
            }

            // Теперь набор данных dataOwns становится тестовой выборкой примеров Свой (обучающие примеры Свой и Чужие удалены из набора данных)
            // Сгенерировали тестовую выборку Чужие
            List<List<double[]>> dataStrangers = DataFactory.GetImgClassesWithCorralatedFeatures(new Random(), 200, 150, 1, 0, 14);

            // Тестируем
            List<int> hemOwn = new List<int>();
            List<int> hemStranger = new List<int>();
            for(int i=0;i<NCTs.Length;i++)
            {
                for(int k=0;k<dataOwns[i].Count;k++)
                {
                    BitArray code = NCTs[i].VerifyImage(dataOwns[i][k]);
                    int hem = 0;
                    for (int h = 0; h < code.Length; h++)
                        if (keys[i][h] != code[h])
                            hem++;
                    hemOwn.Add(hem);
                }
                for (int k = 0; k < dataStrangers.Count; k++)
                {
                    BitArray code = NCTs[i].VerifyImage(dataStrangers[k][0]);
                    int hem = 0;
                    for (int h = 0; h < code.Length; h++)
                        if (keys[i][h] != code[h])
                            hem++;
                    hemStranger.Add(hem);
                }
            }
            // В массивах hemOwn и hemStranger лежат расстояния Хемминга по которым можно сделать выводы об качетсве НКП
    }
}

namespace NCT_framework
{
    // Нейро-корреляционный преобразователь (НКП)

    public class NCT
    {
        public NCT() { }

        // Гиперпараметры
        // _neuronsNumber; // количество нейронов
        // _neuronsInputsNumber; // количество входов нейронов
        // p; // степенной коэффициент (для перехода в мета-пространство Байеса-Минковского), не рекомендуется изменять

        // Закрытые гиперпараметры, которые следует удалять (не сохранять) после обучения
        // _cor_min; // Границы интервалов коррелированности признаков, не рекомендуется существенно изменять
        // _cor_max; // Границы интервалов коррелированности признаков, не рекомендуется существенно изменять
        // _key; // Ключ, связываемый с классом Свой
        // _neuronsAssimetry; // Допустимая разница между количеством нейронов разных типов, не рекомендуется увеличивать стандартное значение
        // _thresholdCoef; // Коэффициент влияния на пороги (используемый при обучении), не рекомендуется существенно изменять
        // minAUC; // Рекомендуемое значение от 0,2 до 0,5
        public void Training(List<double[]> ownImgs, List<double[]> strangerImgs, int featuresCount, BitArray key, int neuronsNumber, int neuronsInputsNumber, Random rand, double minAUC = 0.3, double p = 0.9, double cor_min = -0.5, double cor_max = 0.5, double neuronsAssimetry = 3, double thresholdCoef = 4)
        {
            // Вычисляем нормирующие коэффициенты
            _sx_stranger = new double[featuresCount];
            for (int j = 0; j < featuresCount; j++)
            {
                List<double> cs = new List<double>();
                for (int i = 0; i < strangerImgs.Count; i++)
                    cs.Add(strangerImgs[i][j]);
                _sx_stranger[j] = Statistica.Sx(cs);
            }

            // Переход в мета-пространство Байеса-Минковского
            double[][] owns = owns = new double[ownImgs.Count][];
            double[][] strangers = new double[strangerImgs.Count][];
            for (int i = 0; i < ownImgs.Count; i++)
                owns[i] = Statistica.GetVectorOfNormalizedFeaturesValues(ownImgs[i], _sx_stranger, p);
            for (int i = 0; i < strangerImgs.Count; i++)
                strangers[i] = Statistica.GetVectorOfNormalizedFeaturesValues(strangerImgs[i], _sx_stranger, p);

            // Вычисление матрицы корреляции между признаками для всех образов Свой
            double[][] cor = Statistica.CalcCorrelationMatrix(Statistica.GetCrossSections(owns));

            // Синтезируем НКП
            _tablesIndexes = new List<int>();
            _synapses = new List<MetaFeature[]>();
            _w = new List<double[]>();
            _thresholds = new List<double[]>();
            while (_synapses.Count < neuronsNumber / 2)
            {
                _tablesIndexes.Clear();
                _synapses.Clear();
                _thresholds.Clear();
                int count_minus_neurons = 0;
                int count_plus_neurons = 0;
                int count_minus = 0;
                int count_plus = 0;
                MetaFeature[] neuron_minus = new MetaFeature[neuronsInputsNumber];
                MetaFeature[] neuron_plus = new MetaFeature[neuronsInputsNumber];
                for (int j = 0; j < featuresCount; j++)
                    for (int t = j + 1; t < featuresCount; t++)
                    {
                        if ((count_minus_neurons < neuronsNumber / 2 + neuronsAssimetry) && (count_minus_neurons < neuronsNumber - count_plus_neurons) && (cor[j][t] < cor_min))
                        {
                            MetaFeature meta_f = new MetaFeature();
                            meta_f.j = j;
                            meta_f.t = t;
                            neuron_minus[count_minus] = meta_f;
                            count_minus++;
                            if (count_minus == neuronsInputsNumber)
                            {
                                double[] w = null;
                                w = GetW(neuron_minus, owns, strangers, p);
                                double[] thresholds = GetThresholds_andSetTableIndex(neuron_minus, owns, strangers, p, minAUC, w, thresholdCoef, key, rand);
                                if (thresholds != null)
                                {
                                    _synapses.Add(neuron_minus);
                                    _w.Add(w);
                                    _thresholds.Add(thresholds);
                                    count_minus_neurons++;
                                }
                                neuron_minus = new MetaFeature[neuronsInputsNumber];
                                count_minus = 0;
                            }
                        }
                        else
                        if ((count_plus_neurons < neuronsNumber / 2 + neuronsAssimetry) && (count_plus_neurons < neuronsNumber - count_minus_neurons) && (cor[j][t] > cor_max))
                        {
                            MetaFeature meta_f = new MetaFeature();
                            meta_f.j = j;
                            meta_f.t = t;
                            neuron_plus[count_plus] = meta_f;
                            count_plus++;
                            if (count_plus == neuronsInputsNumber)
                            {
                                double[] w = null;
                                w = GetW(neuron_plus, owns, strangers, p);
                                double[] thresholds = GetThresholds_andSetTableIndex(neuron_plus, owns, strangers, p, minAUC, w, thresholdCoef, key, rand);
                                if (thresholds != null)
                                {
                                    _synapses.Add(neuron_plus);
                                    _w.Add(w);
                                    _thresholds.Add(thresholds);
                                    count_plus_neurons++;
                                }
                                neuron_plus = new MetaFeature[neuronsInputsNumber];
                                count_plus = 0;
                            }
                        }
                    }
                int totalNeurons = count_minus_neurons + count_plus_neurons;
                Console.WriteLine(" Синтезировано: " + count_minus_neurons + "(отр) + " + count_plus_neurons + "(пол) = " + totalNeurons);
                bool ok = false;
                if (count_minus_neurons < neuronsNumber / 2 + neuronsAssimetry)
                    if (cor_min < -0.35)
                    {
                        cor_min = cor_min + 0.05;
                        ok = true;
                    }
                if (count_plus_neurons < neuronsNumber / 2 + neuronsAssimetry)
                    if (cor_max > 0.35)
                    {
                        cor_max = cor_max - 0.05;
                        ok = true;
                    }
                if (!ok) return;
            }
        }

        public BitArray VerifyImage(double[] img, double p = 0.9)
        {
            BitArray res = new BitArray(2 * _synapses.Count);
            double[] realization_norm = null;
            realization_norm = Statistica.GetVectorOfNormalizedFeaturesValues(img, _sx_stranger, p);

            for (int i = 0; i < _synapses.Count; i++)
            {
                double y = GetNeuronOutput(_synapses[i], realization_norm, p, _w[i]);
                bool[] bits = GetNeuronActivation(y, _thresholds[i], _tablesIndexes[i]);
                res[i * 2] = bits[0];
                res[i * 2 + 1] = bits[1];
            }
            return res;
        }

        // Параметры обученного НКП (знания)
        private List<MetaFeature[]> _synapses = new(); // связи корреляционных нейронов с мета-признаками
        private List<int> _tablesIndexes = new(); // номера таблиц преобразований нейронов
        private List<double[]> _thresholds = new(); // пороги нейронов
        private double[] _sx_stranger; // нормирующие коэффициенты признаков (для перехода в мета-пространство Байеса-Минковского)
        private List<double[]> _w; // веса нейронов

        // ТАБЛИЦЫ ПРЕОБРАЗОВАНИЙ откликов нейрона в бинарный код
        static private bool[][][] _tables_patterns = new bool[][][] {
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
            new bool[][] { new bool[] { false, true }, new bool[] { true, true }, new bool[] { false, false }, new bool[] { true, false } } };

        private int GetTableIndex(int rightIntervalIndex, bool[] rightBits, Random rand)
        {
            List<int> indexes = new List<int>();
            for (int i = 0; i < _tables_patterns.Length; i++)
                if (_tables_patterns[i][rightIntervalIndex][0] == rightBits[0] && _tables_patterns[i][rightIntervalIndex][1] == rightBits[1])
                    indexes.Add(i);
            int index = rand.Next(0, indexes.Count);
            return indexes[index];
        }

        // Если возвращает null, то нейрон невозможно создать
        private double[] GetThresholds_andSetTableIndex_inside(double left_own, double right_own, double delta1, double delta2, double delta3, Feature ownFeature, Feature aliensFeature, BitArray key, Random rand)
        {
            double[] res = new double[3];
            int intervalsIndex;
            if (delta2 < 0.1) // 0.1
            { return null;} 

            if (delta2 > 0.4) // 0.4
            {  return null; }

            else
            if (delta1 < 0.1) // 0.1
            {
                res[0] = right_own;
                if (delta3 < 0.6) // 0.6
                {return null; } 

                else
                {
                    intervalsIndex = 0;
                    double right_aliens = aliensFeature.Mx + 4 * aliensFeature.Sx;
                    double delta_aliens = (right_aliens - right_own) / 4;
                    res[1] = res[0] + delta_aliens;
                    res[2] = res[1] + delta_aliens;
                }
            }
            else
            if (delta1 > 0.4)
            {
                if (delta3 < 0.1)
                {
                    if (delta3 + delta2 > 0.4) // 0.4
                    { return null;}   
                    res[2] = left_own;
                    double left_aliens = aliensFeature.Mx - 4 * aliensFeature.Sx;
                    double delta_aliens = (left_own - left_aliens) / 4;
                    res[1] = res[2] - delta_aliens;
                    res[0] = res[1] - delta_aliens;
                    intervalsIndex = 3;
                }
                else
                {
                    double left_aliens = aliensFeature.Mx - 4 * aliensFeature.Sx;
                    double delta_aliens = (left_own - left_aliens) / 3;
                    res[2] = right_own;
                    res[1] = left_own;
                    res[0] = left_own - delta_aliens;
                    intervalsIndex = 2;
                }
            }
            else
            {
                double right_aliens = aliensFeature.Mx + 4 * aliensFeature.Sx;
                double delta_aliens = (right_aliens - right_own) / 3;
                res[0] = left_own;
                res[1] = right_own;
                res[2] = res[1] + delta_aliens;
                intervalsIndex = 1;
            }
            bool[] bits = new bool[2];
            bits[0] = key[_synapses.Count * 2];
            bits[1] = key[_synapses.Count * 2 + 1];
            _tablesIndexes.Add(GetTableIndex(intervalsIndex, bits, rand));
            return res;
        }

        // Если возвращает null, то нейрон невозможно создать
        private double[] GetThresholds_andSetTableIndex(MetaFeature[] neuron, double[][] own, double[][] aliens, double p, double minimalSquare, double[] w, double thresholdCoef_2, BitArray key, Random rand)
        {
            double[] ownOutputs = new double[own.Length];
            double[] aliensOutputs = new double[aliens.Length];
            for (int i = 0; i < own.Length; i++)
                ownOutputs[i] = GetNeuronOutput(neuron, own[i], p, w);
            for (int i = 0; i < aliens.Length; i++)
                aliensOutputs[i] = GetNeuronOutput(neuron, aliens[i], p, w);
            Feature ownFeature = new Feature(ownOutputs);
            Feature aliensFeature = new Feature(aliensOutputs);
            if (minimalSquare < 1)
            {
                double square = Statistica.EvaluateSquare(ownFeature, aliensFeature, 100);
                if (square > minimalSquare)
                {
                    return null;
                }
            }
            double thresholdCoef = 1;
            double left_own = ownFeature.Mx - thresholdCoef_2 * thresholdCoef * ownFeature.Sx;
            double right_own = ownFeature.Mx + thresholdCoef_2 * thresholdCoef * ownFeature.Sx;
            double delta1 = aliensFeature.GetDistributionFunction(left_own);
            double delta = aliensFeature.GetDistributionFunction(right_own);
            double delta2 = delta - delta1;
            while (delta2 < 0.1) // 0.1
            {
                thresholdCoef = thresholdCoef * 1.05;
                left_own = ownFeature.Mx - thresholdCoef_2 * thresholdCoef * ownFeature.Sx;
                right_own = ownFeature.Mx + thresholdCoef_2 * thresholdCoef * ownFeature.Sx;
                delta1 = aliensFeature.GetDistributionFunction(left_own);
                delta = aliensFeature.GetDistributionFunction(right_own);
                delta2 = delta - delta1;
            }

            double delta3 = 1 - delta;
            return GetThresholds_andSetTableIndex_inside(left_own, right_own, delta1, delta2, delta3, ownFeature, aliensFeature, key, rand);
        }

        private double[][] GetMetaFeaturesOfSecondOrder(MetaFeature[] neuron, double[][] normalizedFetures, double p)
        {
            double[][] res = new double[normalizedFetures.Length][];
            for (int j = 0; j < normalizedFetures.Length; j++)
            {
                double[] metaFeatures = new double[neuron.Length];
                double my = Math.Abs(Math.Abs(normalizedFetures[j][neuron[0].j]) - Math.Abs(normalizedFetures[j][neuron[0].t]));
                metaFeatures[0] = my;
                for (int i = 1; i < neuron.Length; i++)
                {
                    metaFeatures[i] = Math.Abs(Math.Abs(normalizedFetures[j][neuron[i].j]) - Math.Abs(normalizedFetures[j][neuron[i].t]));
                    my = Statistica.Mx_rct(my, metaFeatures[i], i + 1);
                }
                for (int i = 0; i < neuron.Length; i++)
                    metaFeatures[i] = Math.Pow(metaFeatures[i] - my, 2);
                res[j] = metaFeatures;
            }
            return res;
        }

        private double[] GetW(MetaFeature[] neuron, double[][] own, double[][] aliens, double p)
        {
            double[][] own_meta = GetMetaFeaturesOfSecondOrder(neuron, own, p);
            double[][] aliens_meta = GetMetaFeaturesOfSecondOrder(neuron, aliens, p);
            double[] res = new double[neuron.Length];
            for (int i = 0; i < neuron.Length; i++)
            {
                double m_own = own_meta[0][i];
                m_own = Statistica.Mx_rct(m_own, own_meta[1][i], 2);
                double d_own = (Math.Pow(own_meta[0][i] - m_own, 2) + Math.Pow(own_meta[1][i] - m_own, 2)) / 2;
                for (int j = 2; j < own_meta.Length; j++)
                {
                    m_own = Statistica.Mx_rct(m_own, own_meta[j][i], j + 1);
                    d_own = Statistica.Dx_rct(d_own, m_own, own_meta[j][i], j + 1);
                }
                double m_aliens = aliens_meta[0][i];
                m_aliens = Statistica.Mx_rct(m_aliens, aliens_meta[1][i], 2);
                double d_aliens = (Math.Pow(aliens_meta[0][i] - m_aliens, 2) + Math.Pow(aliens_meta[1][i] - m_aliens, 2)) / 2;
                for (int j = 2; j < aliens_meta.Length; j++)
                {
                    m_aliens = Statistica.Mx_rct(m_aliens, aliens_meta[j][i], j + 1);
                    d_aliens = Statistica.Dx_rct(d_aliens, m_aliens, aliens_meta[j][i], j + 1);
                }
                d_own = Math.Sqrt(d_own);
                d_aliens = Math.Sqrt(d_aliens);
                res[i] = Math.Abs(m_own - m_aliens) / (d_own * d_aliens);
            }
            return res;
        }

        private double GetNeuronOutput(MetaFeature[] neuron, double[] normalizedFetureVector, double p, double[] w)
        {
            double[] metaFeatures = new double[neuron.Length];
            double my = Math.Abs(Math.Abs(normalizedFetureVector[neuron[0].j]) - Math.Abs(normalizedFetureVector[neuron[0].t]));
            metaFeatures[0] = my;
            for (int i = 1; i < neuron.Length; i++)
            {
                metaFeatures[i] = Math.Abs(Math.Abs(normalizedFetureVector[neuron[i].j]) - Math.Abs(normalizedFetureVector[neuron[i].t]));
                my = Statistica.Mx_rct(my, metaFeatures[i], i + 1);
            }
            double my___ = Math.Pow(metaFeatures[0] - my, 2) * w[0];
            for (int i = 1; i < neuron.Length; i++)
            {
                metaFeatures[i] = Math.Pow(metaFeatures[i] - my, 2) * w[i];
                my___ = Statistica.Mx_rct(my___, metaFeatures[i], i + 1);
            }
            my___ = Math.Sqrt(my___);
            return my___;
        }

        private bool[] GetNeuronActivation(double y, double[] thresholds, int tableIndex)
        {
            if (y < thresholds[0])
                return _tables_patterns[tableIndex][0];
            if ((thresholds[0] <= y) && (y < thresholds[1]))
                return _tables_patterns[tableIndex][1];
            if ((thresholds[1] <= y) && (y < thresholds[2]))
                return _tables_patterns[tableIndex][2];
            return _tables_patterns[tableIndex][3];
        }
    
        public void SerializeToWriter(BinaryWriter writer)
        {
            // Сохраняем sxstranger
            writer.Write(_sx_stranger.Length);
            foreach (var sx in _sx_stranger)
                writer.Write(sx);

            // Сохраняем synapses, thresholds, w, tablesIndexes
            writer.Write(_synapses.Count);
            for (int i = 0; i < _synapses.Count; i++)
            {
                writer.Write(_synapses[i].Length);
                foreach (var mf in _synapses[i])
                {
                    writer.Write(mf.j);
                    writer.Write(mf.t);
                }
                writer.Write(_w[i].Length);
                foreach (var wi in _w[i])
                    writer.Write(wi);
                writer.Write(_thresholds[i].Length);
                foreach (var th in _thresholds[i])
                    writer.Write(th);
                writer.Write(_tablesIndexes[i]);
            }
        }

        public static NCT DeserializeFromReader(BinaryReader reader)
        {
            var nct = new NCT();
            
            // Восстанавливаем sxstranger
            int sxLen = reader.ReadInt32();
            nct._sx_stranger = new double[sxLen];
            for (int i = 0; i < sxLen; i++)
                nct._sx_stranger[i] = reader.ReadDouble();

            // Восстанавливаем synapses и прочее
            int synCount = reader.ReadInt32();
            nct._synapses = new List<MetaFeature[]>();
            nct._w = new List<double[]>();
            nct._thresholds = new List<double[]>();
            nct._tablesIndexes = new List<int>();

            for (int i = 0; i < synCount; i++)
            {
                int metaLen = reader.ReadInt32();
                var mf = new MetaFeature[metaLen];
                for (int j = 0; j < metaLen; j++)
                {
                    mf[j].j = reader.ReadInt32();
                    mf[j].t = reader.ReadInt32();
                }
                nct._synapses.Add(mf);

                int wLen = reader.ReadInt32();
                var w = new double[wLen];
                for (int j = 0; j < wLen; j++)
                    w[j] = reader.ReadDouble();
                nct._w.Add(w);

                int thLen = reader.ReadInt32();
                var th = new double[thLen];
                for (int j = 0; j < thLen; j++)
                    th[j] = reader.ReadDouble();
                nct._thresholds.Add(th);

                nct._tablesIndexes.Add(reader.ReadInt32());
            }

            return nct;
        }
    }


    // Признак, значения которого подчинены нормальному закону распределения
    public class Feature
    {
        private double _mx;
        private double _dx;
        private double _sx;

        private static double Erf(double x)
        {
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = Math.Abs(x);
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
            return sign * y;
        }

        public Feature(double[] featureValues)
        {
            _mx = Statistica.Mx(featureValues);
            _dx = Statistica.Dx(featureValues, _mx);
            _sx = Math.Sqrt(_dx);
        }

        public double Mx
        { get { return _mx; } }

        public double Dx
        { get { return _dx; } }

        public double Sx
        { get { return _sx; } }

        public double GetDensityOfProb(double x)
        { return Math.Exp(-(((x - _mx) * (x - _mx)) / (2 * _dx))) / (Math.Sqrt(_dx) * Math.Sqrt(2 * Math.PI)); }

        public double GetDistributionFunction(double x)
        { return (1 + Erf((x - _mx) / Math.Sqrt(2 * _dx))) / 2; }

        public double GetMinimumLimit()
        { return _mx - (4 * _sx); }

        public double GetMaximumLimit()
        { return _mx + (4 * _sx); }
    }


    // Мета-признак
    public struct MetaFeature
    {
        public int j; // номер первого признака в паре
        public int t; // номер второго признака в паре
    }

    // Фабрика ключей
    public static class KeyFactory
    {
        public static BitArray GetKey(Random random, int neuronsNumber, int bitsPerNeuron = 2)
        {
            BitArray key = new BitArray(neuronsNumber * bitsPerNeuron);
            for (int i = 0; i < neuronsNumber * bitsPerNeuron; i++)
                if (random.Next(0, bitsPerNeuron) == 0)
                    key[i] = true;
                else
                    key[i] = false;
            return key;
        }
    }

    // Фабрика данных
    public static class DataFactory
    {
        //Считать признаки изображений из файла
        public static List<List<double[]>> ExtractFeaturesFromFile(string filePath, int imageClassesCount, int imgPerClass, int featureCount)
        {
            List<List<double[]>> data = new List<List<double[]>>();
            for (int i = 0; i < imageClassesCount; i++)
            {
                data.Add(new List<double[]>());
                for (int j = 0; j < imgPerClass; j++)
                    data[i].Add(new double[featureCount]);
            }
            try
            {
                var lines = File.ReadAllLines(filePath);
                for (int lineIndex = 0; lineIndex < lines.Length; lineIndex++)
                {
                    var line = lines[lineIndex];
                    int classIndex = lineIndex / imgPerClass;
                    int imageIndex = lineIndex % imgPerClass;

                    var features = line.Split(',')
                                    .Take(featureCount)
                                    .Select(s => double.Parse(s, CultureInfo.InvariantCulture))
                                    .ToArray();
                    if (classIndex < imageClassesCount)
                        data[classIndex][imageIndex] = features;
                } 
            }
            catch (Exception ex)
            {
                Console.WriteLine("Ошибка при чтении файла: " + ex.Message);
            }
            return data;
        }

        // Сгенерировать данные (Чем больше разница между mx_min и mx_max и меньше классов, тем информативнее признаки)
        public static List<List<double[]>> GetImgClassesWithCorralatedFeatures(Random r, int featureCount, int imageClassesCount, int imgPerClass, double mx_min, double mx_max)
        {
            List<List<double[]>> data = new List<List<double[]>>();
            List<List<double[]>> res = new List<List<double[]>>();
            double dif = mx_max - mx_min;
            for (int i = 0; i < imageClassesCount; i++)
            {
                data.Add(new List<double[]>());
                res.Add(new List<double[]>());
                bool dirrectInverse = false;
                for (int j = 0; j < featureCount; j++)
                {
                    data[i].Add(new double[imgPerClass]);
                    double mx = r.NextDouble() * dif + mx_min;
                    for (int k = 0; k < data[i][j].Length; k++)
                        data[i][j][k] = GenerateFeatureValue(mx, 1, r);
                    Array.Sort(data[i][j]);
                        if (dirrectInverse)
                            Array.Reverse(data[i][j]);
                    dirrectInverse = !dirrectInverse;
                }
                for (int k = 0; k < imgPerClass; k++)
                {
                    res[i].Add(new double[featureCount]);
                    for (int j = 0; j < featureCount; j++)
                        res[i][k][j] = data[i][j][k];
                }
            }
            data = null;
            return res;
        }

        // Сгенерировать значение признака
        public static double GenerateFeatureValue(double mx, double sx, Random r)
        {
            double summa = 0;
            for (int i = 0; i < 12; i++)
                summa += r.NextDouble();
            summa = summa - 6;
            return (summa * sx) + mx;
        }
    }


    public static class Statistica
    {
        // Мат. ожидание, как среднее арифметическое
        public static double Mx(List<double> x)
        {
            double result = 0;
            for (int i = 0; i < x.Count; i++)
                result = result + x[i];
            return result / x.Count;
        }
        public static double Mx(double[] x)
        {
            double result = 0;
            for (int i = 0; i < x.Length; i++)
                result = result + x[i];
            return result / x.Length;
        }

        // Среднеквадратичное отклонение
        public static double Sx(double[] x, double mx)
        { return Math.Sqrt(Dx(x, mx)); }
        public static double Sx(List<double> x)
        { return Math.Sqrt(Dx(x)); }

        // Медиана
        public static double Median(List<double> x) {
            var arr = x.ToArray(); Array.Sort(arr);
            int n = arr.Length;
            return (n % 2 == 1) ? arr[n/2] : 0.5*(arr[n/2 - 1] + arr[n/2]);
        }
        // Отклонение по медиане
        public static double MAD(List<double> x) {
            double med = Median(x);
            List<double> dev = new List<double>(x.Count);
            for (int i=0;i<x.Count;i++) dev.Add(Math.Abs(x[i] - med));
            return Median(dev);
        }


        // Дисперсия
        public static double Dx(List<double> x)
        {
            double[] result = new double[x.Count];
            double m = Mx(x);
            for (int i = 0; i < x.Count; i++)
                result[i] = (x[i] - m) * (x[i] - m);
            return Mx(result);
        }
        public static double Dx(List<double> x, double mx)
        {
            double[] result = new double[x.Count];
            for (int i = 0; i < x.Count; i++)
                result[i] = (x[i] - mx) * (x[i] - mx);
            return Mx(result);
        }
        public static double Dx(double[] x)
        {
            double[] result = new double[x.Length];
            double m = Mx(x);
            for (int i = 0; i < x.Length; i++)
                result[i] = (x[i] - m) * (x[i] - m);
            return Mx(result);
        }
        public static double Dx(double[] x, double mx)
        {
            double[] result = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
                result[i] = (x[i] - mx) * (x[i] - mx);
            return Mx(result);
        }

        // Рекурентное вычисление мат. ожидания
        public static double Mx_rct(double mx_prev, double x, int n)
        { return (((double)(n - 1) / n) * mx_prev) + (((double)1 / n) * x); }

        // Рекурентное вычисление дисперсии
        public static double Dx_rct(double disx_prev, double mx, double x, int n)
        { return (((double)(n - 2) / (n - 1)) * disx_prev) + (((double)1 / (n - 1)) * ((x - mx) * (x - mx))); }

        // Вычислить площадь пересечения функций плотности вероятности признака для класса Свой и Чужие
        static public double EvaluateSquare(Feature own, Feature stranger, uint accuracy)
        {
            double min = own.GetMinimumLimit();
            double max = own.GetMaximumLimit();
            double min2 = stranger.GetMinimumLimit();
            double max2 = stranger.GetMaximumLimit();
            if (min > min2) min = min2;
            if (max < max2) max = max2;
            double e = 0;
            double step = (max - min) / accuracy;
            if (step == 0)
                return 1;
            double st = min;
            int counter = 0;
            while ((st < max) && (counter < 10000)) // костыль для ебаного касяка .Net, я охуел когда это произошло
            {
                double t_i = own.GetDensityOfProb(st);
                double t_j = stranger.GetDensityOfProb(st);
                double T1 = t_j > t_i ? t_i : t_j;
                if ((!Double.IsNaN(T1)) && (!Double.IsInfinity(T1)))
                    e += T1 * step;
                st += step;
                counter++;
            }
            if (e == 0) e = e + ((double)1 / (100 * accuracy));
            return e;
        }

        // Момент корреляции
        public static double MomentCor(double[] x, double[] y)
        {
            double[] Mul = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
                Mul[i] = x[i] * y[i];
            return Mx(Mul) - (Mx(x) * Mx(y));
        }

        // Коэффициент корреляции
        public static double CoefCor(double[] x, double[] y)
        { return MomentCor(x, y) / (Math.Sqrt(Dx(x)) * Math.Sqrt(Dx(y))); }

        // Вычисление корреляционной матрицы между сечениями всех пар признаков (сечение - совокупность значений признаков)
        static public double[][] CalcCorrelationMatrix(double[][] featuresValues)
        {
            double[][] res = new double[featuresValues.Length][];
            for (ushort i = 0; i < featuresValues.Length; i++)
                res[i] = new double[featuresValues.Length];
            for (ushort i = 0; i < featuresValues.Length; i++)
            {
                res[i][i] = 1;
                for (ushort j = (ushort)(i + 1); j < featuresValues.Length; j++)
                {
                    res[i][j] = Statistica.CoefCor(featuresValues[j], featuresValues[i]);
                    res[j][i] = res[i][j];
                }
            }
            return res;
        }

        // Преобразовать массив образов в массив сечений (сечение - совокупность значений признаков)
        static public double[][] GetCrossSections(double[][] realizations)
        {
            double[][] res = new double[realizations[0].Length][];
            for (int j = 0; j < realizations[0].Length; j++)
            {
                res[j] = new double[realizations.Length];
                for (int i = 0; i < realizations.Length; i++)
                    res[j][i] = realizations[i][j];
            }
            return res;
        }

        // Нормировать вектор признаков относительно среднеквадратичных отклонений класса Чужие
        static public double[] GetVectorOfNormalizedFeaturesValues(double[] allFeaturesValues, double[] sx, double p)
        {
            double[] res = new double[allFeaturesValues.Length];
            for (int j = 0; j < allFeaturesValues.Length; j++)
                res[j] = Math.Pow(Math.Abs(allFeaturesValues[j]) / sx[j], p);
            return res;
        }
    }
}
