import json
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass

from logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetaFeature:
    j: int
    t: int
    
    def __hash__(self):
        return hash((self.j, self.t))
    
    def __eq__(self, other):
        return self.j == other.j and self.t == other.t

class CorrelationGraphBuilder:
    
    def __init__(self, meta_path: str, nct_index: int = 0):
        self.meta_path = Path(meta_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(f"File not found: {self.meta_path}")
         
        self.nct_index = nct_index
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        self.nct_data = meta['ncts'][self.nct_index]
        self.n_neurons = meta['neurons_count']
        self.n_features = meta['feature_count']
        
        # Структуры данных для анализа
        self.feature_partners = {}      # feature -> set(feature_t)
        self.feature_importance = {}    # feature -> importance score
        self.feature_degree = {}        # feature -> degree
        self.neurons_by_feature = {}    # feature -> list(neuron_idx)
        
        self._build_graph()
    

    def _build_graph(self):
        
        synapses = self.nct_data.get('synapses', [])

        unique_pairs = set()
        temp_degree = {}       

        for neuron_idx, neuron_synapses in enumerate(synapses):
            for feature_pair in neuron_synapses:

                feature_i = int(feature_pair[0])
                feature_t = int(feature_pair[1])

                for feature_id in [feature_i, feature_t]:
                    if feature_id not in self.neurons_by_feature:
                        self.neurons_by_feature[feature_id] = set()
                    self.neurons_by_feature[feature_id].add(neuron_idx)
                
                if feature_i > feature_t:
                    feature_i, feature_t = feature_t, feature_i
                
                pair_key = (feature_i, feature_t)

                if pair_key not in unique_pairs:
                    unique_pairs.add(pair_key)
                    temp_degree[feature_i] = temp_degree.get(feature_i, 0) + 1
                    temp_degree[feature_t] = temp_degree.get(feature_t, 0) + 1
        
        for (feature_i, feature_t) in unique_pairs:
            
            degree_i = temp_degree.get(feature_i, 0)
            degree_t = temp_degree.get(feature_t, 0)
            
            # владелец пары
            if degree_i > degree_t:
                owner = feature_i
                partner = feature_t
            elif degree_t > degree_i:
                owner = feature_t
                partner = feature_i
            else:
                owner = feature_i
                partner = feature_t
            
            if owner not in self.feature_partners:
                self.feature_partners[owner] = {}
            
            partner_degree = temp_degree.get(partner, 0)
            self.feature_partners[owner][partner] = partner_degree

        for owner in self.feature_partners:
            self.feature_degree[owner] = len(self.feature_partners[owner])

        max_degree = max(self.feature_degree.values())
        
        for feature_id, degree in self.feature_degree.items():
            self.feature_importance[feature_id] = min(1.0, degree / max_degree)
        
        print(f"[DONE] Граф построен:")
        print(f"  - Родительских признаков: {len(self.feature_partners)}")
        print(f"  - Наибольшее кол-во партнеров у признака: {max_degree}")


    def save_graph_to_json(self, output_path: str = "model/graph.json") -> bool:

        all_feature_ids = set(list(self.feature_degree.keys()) + 
                            list(self.neurons_by_feature.keys()))
        
        sorted_feature_ids = sorted(
            all_feature_ids,
            key=lambda fid: self.feature_importance.get(fid, 0.0),
            reverse=True  
        )

        features_dict = {}
        for feature_id in sorted_feature_ids:

            partners = self.feature_partners.get(feature_id, {})
            degree = len(partners)

            sorted_partners = sorted(
                partners.items(),
                key=lambda x: self.feature_importance.get(x[0], 0.0), 
                reverse=True 
            )
            
            features_dict[str(feature_id)] = {
                "degree": degree,
                "importance": round(self.feature_importance.get(feature_id, 0.0), 4),
                "neurons": sorted(list(self.neurons_by_feature.get(feature_id, set()))),
                "neurons_count": len(self.neurons_by_feature.get(feature_id, set())),
                "partners": {
                    str(partner_id): round(self.feature_importance.get(partner_id, 0.0), 4)
                    for partner_id, weight in sorted_partners
                },
            }
        
        graph_data = {
            "nct_index": self.nct_index,
            "features": features_dict,
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(graph_data, f, indent=2, ensure_ascii=False)
            json.dump(
                graph_data,
                f,
                indent=2,            
                ensure_ascii=False,    
            )
        
        file_size_kb = output_path.stat().st_size / 1024
        total_pairs = sum(len(p) for p in self.feature_partners.values())
        
        print(f"  - Граф сохранён в {output_path}")
        print(f"  - Размер файла: {file_size_kb:.1f} KB")
        
        return True
        

    def get_feature_importance(self, feature_id: int) -> float:
        return self.feature_importance.get(feature_id, 0.0)
    

    def get_partners(self, feature_id: int, top_n: int = None) -> List[int]:
        if feature_id not in self.feature_partners:
            return []
        
        partners = list(self.feature_partners[feature_id])
        if top_n:
            partners = partners[:top_n]
        
        return partners
    


    # def get_feature_stats(self, feature_id: int) -> Dict:
    #     return {
    #         'feature_id': feature_id,
    #         'importance': self.get_feature_importance(feature_id),
    #         'occurrences': self.feature_degree.get(feature_id, 0),
    #         'neurons_involved': len(self.neurons_by_feature.get(feature_id, [])),
    #         'partners_count': len(self.feature_partners.get(feature_id, set())),
    #         'top_partners': self.get_partners(feature_id, top_n=5),
    #     }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Построение графа корреляций")
    parser.add_argument(
        "--meta-path",
        type=str,
        default="../model/meta.json",
        help="Путь к meta.json"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../model/graph.json",
        help="Путь к graph.json"
    )
    parser.add_argument(
        "--nct-index",
        type=int,
        default=0,
        help="Индекс NCT в массиве"
    )
    
    args = parser.parse_args()
    builder = CorrelationGraphBuilder(args.meta_path)
    builder.save_graph_to_json(args.output_path)
