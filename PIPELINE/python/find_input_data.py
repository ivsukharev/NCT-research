import argparse
import json
import numpy as np
from pathlib import Path

def select_best_alien_candidate(target_vector: np.ndarray, 
                                alien_matrix: np.ndarray, 
                                graph_data: dict) -> np.ndarray:
    """
    Рассчитывает скор для матрицы 'чужих' изображений.
    :return: Массив Score для каждого чужого изображения.
    """
    # 1. Считаем дельты (стандартное отклонение) по выборке нарушителя
    delta = np.std(alien_matrix, axis=0)
    # Защита от деления на ноль для константных признаков
    delta[delta == 0] = 1e-8 
    
    # 2. Нелинейная трансформация всех признаков в пространство НКП
    v_target = np.power(np.abs(target_vector / delta), 0.9)
    v_aliens = np.power(np.abs(alien_matrix / delta), 0.9)
    
    M_aliens = alien_matrix.shape[0]
    scores = np.zeros(M_aliens)
    
    # 3. Вычисляем расстояние в пространстве мета-признаков
    features_dict = graph_data.get('features', {})
    
    for t_str, parent_info in features_dict.items():
        t = int(t_str)
        I_t = parent_info.get('importance', 0.0)
        
        for j_str, I_j in parent_info.get('partners', {}).items():
            j = int(j_str)
            
            # Мета-признак жертвы
            meta_target = abs(v_target[t] - v_target[j])
            
            # Мета-признаки всех чужих
            meta_aliens = np.abs(v_aliens[:, t] - v_aliens[:, j])
            
            # Взвешиваем разницы (I_t * I_j) и суммируем
            weight = I_t * I_j
            scores += weight * np.abs(meta_aliens - meta_target)
            
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к *_3d_data.csv (без заголовка)')
    parser.add_argument('--graph-json', required=True, help='Путь к graph.json')
    parser.add_argument('--target-class', type=int, required=True, help='Индекс целевого класса (0-based)')
    parser.add_argument('--target-photo', type=int, default=11, help='Номер целевого фото (1-based)')
    parser.add_argument('--n-classes', type=int, default=200)
    parser.add_argument('--n-per-class', type=int, default=14)
    parser.add_argument('--n-foreign', type=int, default=30, help='Количество чужих персон для отбора')
    parser.add_argument('--n-foreign-photos', type=int, default=2, help='Количество фотографий на каждого чужого')
    args = parser.parse_args()

    # Загружаем данные признаков
    print(f"[*] Чтение датасета: {args.input}")
    data = np.loadtxt(args.input, delimiter=',')
    n_samples, n_features = data.shape
    expected = args.n_classes * args.n_per_class
    assert n_samples == expected, f"Датасет не совпадает: {n_samples} строк вместо {expected}"

    # Загружаем граф
    print(f"[*] Инференс графа: {args.graph_json}")
    with open(args.graph_json, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Определяем индексы
    target_img_idx = args.target_photo - 1
    target_global_idx = args.target_class * args.n_per_class + target_img_idx
    target_vector = data[target_global_idx]

    # Собираем "чужих" кандидатов
    alien_global_indices = []
    foreign_classes = [c for c in range(args.n_classes) if c != args.target_class][:args.n_foreign]
    
    for c in foreign_classes:
        for p in range(args.n_foreign_photos):
            alien_global_indices.append(c * args.n_per_class + p)
        
    alien_global_indices = np.array(alien_global_indices)
    alien_matrix = data[alien_global_indices]

    print(f"[*] Целевое изображение: Person {args.target_class+1}, Photo {args.target_photo}")
    print(f"[*] Количество доступных 'чужих' персон: {args.n_foreign}, по {args.n_foreign_photos} фото на каждого (всего {len(alien_global_indices)} фото)\n")
    
    # ------------------ Запуск алгоритма ------------------
    scores = select_best_alien_candidate(target_vector, alien_matrix, graph_data)
    # ------------------------------------------------------
    
    # Вывод Топ-5 изображений
    sorted_idx = np.argsort(scores)
    samples_arg = []
    top_k = 10
    
    print(f"[+] Топ-{top_k} лучших чужих изображений (минимальный Score):")
    for i in range(min(top_k, len(scores))):
        idx_in_alien_arr = sorted_idx[i]
        global_idx = alien_global_indices[idx_in_alien_arr]
        
        person = global_idx // args.n_per_class
        photo = global_idx % args.n_per_class
        
        print(f"  {i+1}. Person: {person+1}, Photo: {photo+1} | Score: {scores[idx_in_alien_arr]:.4f}")
        samples_arg.append(f"{person+1}:{photo+1}")

    print("\n[i] Вы можете подставить найденные изображения в аргумент вызова сборки датасета для атаки.")
    print("---------------------------------------")
    print(f"make build-attack-data SAMPLES=\"{','.join(samples_arg)}\"")
    print("---------------------------------------")

if __name__ == '__main__':
    main()
