# evaluation_system.py - 桌游推荐系统评估框架
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import logging
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, recommender_system, data_df):
        self.recommender = recommender_system
        self.df = data_df
        self.evaluation_results = {}
        
    def create_synthetic_users(self, n_users: int = 100) -> List[Dict]:
        """创建合成用户档案用于评估"""
        synthetic_users = []
        
        # 定义不同类型的玩家画像
        user_profiles = [
            {
                'name': 'strategy_expert',
                'preferred_mechanics': ['strategy', 'building', 'territory'],
                'preferred_domains': ['Strategy Games'],
                'complexity_range': (3.0, 4.5),
                'playtime_range': (90, 240),
                'player_count': (2, 4)
            },
            {
                'name': 'family_gamer',
                'preferred_mechanics': ['luck', 'cards', 'reaction'],
                'preferred_domains': ['Family Games'],
                'complexity_range': (1.0, 2.5),
                'playtime_range': (30, 90),
                'player_count': (2, 6)
            },
            {
                'name': 'party_host',
                'preferred_mechanics': ['social', 'reaction', 'roleplay'],
                'preferred_domains': ['Party Games'],
                'complexity_range': (1.0, 2.0),
                'playtime_range': (15, 60),
                'player_count': (4, 8)
            },
            {
                'name': 'coop_enthusiast',
                'preferred_mechanics': ['cooperation', 'roleplay', 'cards'],
                'preferred_domains': ['Thematic Games'],
                'complexity_range': (2.0, 3.5),
                'playtime_range': (60, 180),
                'player_count': (2, 4)
            },
            {
                'name': 'solo_player',
                'preferred_mechanics': ['strategy', 'building', 'cards'],
                'preferred_domains': ['Strategy Games'],
                'complexity_range': (2.5, 4.0),
                'playtime_range': (30, 120),
                'player_count': (1, 2)
            }
        ]
        
        # 为每个用户画像生成多个变体
        for i in range(n_users):
            profile = user_profiles[i % len(user_profiles)]
            
            # 添加随机变化
            complexity = np.random.uniform(profile['complexity_range'][0], profile['complexity_range'][1])
            playtime = int(np.random.uniform(profile['playtime_range'][0], profile['playtime_range'][1]))
            
            # 从高评分游戏中选择作为"喜欢的游戏"（ground truth）
            liked_games = self._select_liked_games_for_profile(profile, n_games=10)
            
            synthetic_user = {
                'user_id': f'synthetic_{i}',
                'profile_type': profile['name'],
                'preferences': {
                    'selectedMechanics': profile['preferred_mechanics'],
                    'selectedDomains': profile['preferred_domains'],
                    'gameSettings': {
                        'players': f"{profile['player_count'][0]}-{profile['player_count'][1]}",
                        'time': str(playtime),
                        'age': '12',
                        'complexity': str(round(complexity, 1))
                    }
                },
                'liked_games': liked_games,  # Ground truth
                'classic_games': self._select_classic_games_for_profile(liked_games, n_games=3)
            }
            
            synthetic_users.append(synthetic_user)
            
        return synthetic_users
    
    def _select_liked_games_for_profile(self, profile: Dict, n_games: int = 10) -> List[int]:
        """为用户画像选择合适的游戏作为ground truth"""
        # 筛选符合用户画像的游戏
        filtered_df = self.df[
            (self.df['Complexity'] >= profile['complexity_range'][0]) &
            (self.df['Complexity'] <= profile['complexity_range'][1]) &
            (self.df['Play Time'] >= profile['playtime_range'][0]) &
            (self.df['Play Time'] <= profile['playtime_range'][1]) &
            (self.df['Min Players'] <= profile['player_count'][0]) &
            (self.df['Max Players'] >= profile['player_count'][1]) &
            (self.df['Rating Average'] >= 7.0) &
            (self.df['Users Rated'] >= 1000)
        ].copy()
        
        # 根据机制偏好进一步筛选
        mechanism_scores = []
        for idx, game in filtered_df.iterrows():
            score = 0
            game_mechanics = game.get('Mechanics', [])
            if isinstance(game_mechanics, list):
                # 检查游戏机制是否匹配用户偏好
                for mech_category in profile['preferred_mechanics']:
                    if mech_category in self.recommender.mechanism_mapping:
                        category_mechanics = self.recommender.mechanism_mapping[mech_category]
                        if any(mech in category_mechanics for mech in game_mechanics):
                            score += 1
            mechanism_scores.append(score)
        
        filtered_df['mechanism_score'] = mechanism_scores
        
        # 选择得分最高的游戏
        top_games = filtered_df.nlargest(n_games * 2, ['mechanism_score', 'Rating Average'])
        
        # 随机选择以增加多样性
        if len(top_games) >= n_games:
            selected_indices = np.random.choice(top_games.index, size=n_games, replace=False)
            return selected_indices.tolist()
        else:
            return top_games.index.tolist()
    
    def _select_classic_games_for_profile(self, liked_games: List[int], n_games: int = 3) -> List[str]:
        """从喜欢的游戏中选择作为经典游戏输入"""
        selected_games = []
        for game_id in liked_games[:n_games]:
            if game_id in self.df.index:
                game_name = self.df.loc[game_id, 'Name']
                selected_games.append(game_name)
        return selected_games
    
    def calculate_ndcg(self, recommended_ids: List[int], relevant_ids: List[int], k: int = 12) -> float:
        """
        计算NDCG@k (Normalized Discounted Cumulative Gain)
        
        NDCG是推荐系统中最重要的评估指标，它考虑了：
        1. 推荐结果的相关性
        2. 推荐结果的排序质量
        """
        # 创建相关性分数（基于游戏是否在ground truth中）
        relevance_scores = []
        for i, game_id in enumerate(recommended_ids[:k]):
            if game_id in relevant_ids:
                # 根据在ground truth中的位置给予不同分数
                rel_position = relevant_ids.index(game_id)
                relevance = 1.0 / (rel_position + 1)  # 位置越靠前，相关性越高
            else:
                relevance = 0.0
            relevance_scores.append(relevance)
        
        # 计算DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 因为位置从1开始
        
        # 计算理想的DCG (IDCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        # 计算NDCG
        if idcg == 0:
            return 0.0
        else:
            return dcg / idcg
    
    def calculate_precision_recall(self, recommended_ids: List[int], relevant_ids: List[int], k: int = 12) -> Tuple[float, float]:
        """计算Precision@k和Recall@k"""
        recommended_set = set(recommended_ids[:k])
        relevant_set = set(relevant_ids)
        
        # 计算交集
        hits = len(recommended_set.intersection(relevant_set))
        
        # Precision: 推荐中相关的比例
        precision = hits / k if k > 0 else 0.0
        
        # Recall: 相关中被推荐的比例
        recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        return precision, recall
    
    def calculate_hit_rate(self, recommended_ids: List[int], relevant_ids: List[int], k: int = 12) -> float:
        """计算Hit Rate@k - 至少有一个相关推荐的比例"""
        recommended_set = set(recommended_ids[:k])
        relevant_set = set(relevant_ids)
        
        return 1.0 if len(recommended_set.intersection(relevant_set)) > 0 else 0.0
    
    def calculate_mrr(self, recommended_ids: List[int], relevant_ids: List[int]) -> float:
        """计算MRR (Mean Reciprocal Rank) - 第一个相关推荐的位置倒数"""
        for i, game_id in enumerate(recommended_ids):
            if game_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_diversity(self, recommended_games: List[Dict]) -> float:
        """计算推荐结果的多样性（基于游戏机制）"""
        if len(recommended_games) < 2:
            return 0.0
        
        # 收集所有游戏的机制
        all_mechanics = []
        for game in recommended_games:
            mechanics = game.get('mechanics', [])
            if mechanics:
                all_mechanics.append(set(mechanics))
        
        # 计算两两之间的Jaccard距离
        diversity_scores = []
        for i in range(len(all_mechanics)):
            for j in range(i + 1, len(all_mechanics)):
                # Jaccard距离 = 1 - Jaccard相似度
                intersection = len(all_mechanics[i].intersection(all_mechanics[j]))
                union = len(all_mechanics[i].union(all_mechanics[j]))
                if union > 0:
                    jaccard_distance = 1 - (intersection / union)
                    diversity_scores.append(jaccard_distance)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def calculate_coverage(self, all_recommendations: List[List[int]]) -> float:
        """计算推荐系统的覆盖率"""
        # 所有被推荐过的游戏
        recommended_games = set()
        for rec_list in all_recommendations:
            recommended_games.update(rec_list)
        
        # 所有可推荐的游戏（评分>=6.5，评价人数>=100）
        recommendable_games = self.df[
            (self.df['Rating Average'] >= 6.5) &
            (self.df['Users Rated'] >= 100)
        ]
        
        coverage = len(recommended_games) / len(recommendable_games)
        return coverage
    
    def evaluate_system(self, n_synthetic_users: int = 100) -> Dict[str, Any]:
        """
        全面评估推荐系统
        
        返回包含多个评估指标的字典
        """
        logger.info(f"开始评估推荐系统，创建{n_synthetic_users}个合成用户...")
        
        # 创建合成用户
        synthetic_users = self.create_synthetic_users(n_synthetic_users)
        
        # 初始化评估指标
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        hit_rates = []
        mrr_scores = []
        diversity_scores = []
        all_recommendations = []
        
        # 对每个合成用户进行评估
        for user in synthetic_users:
            try:
                # 添加经典游戏到偏好中
                user['preferences']['selectedGames'] = user['classic_games']
                
                # 获取推荐
                recommendations = self.recommender.get_enhanced_recommendations(
                    user['preferences'], 
                    N=20
                )
                
                if not recommendations:
                    continue
                
                # 提取推荐的游戏ID
                recommended_ids = [rec['id'] for rec in recommendations]
                all_recommendations.append(recommended_ids)
                
                # 计算各项指标
                ndcg = self.calculate_ndcg(recommended_ids, user['liked_games'], k=12)
                precision, recall = self.calculate_precision_recall(recommended_ids, user['liked_games'], k=12)
                hit_rate = self.calculate_hit_rate(recommended_ids, user['liked_games'], k=12)
                mrr = self.calculate_mrr(recommended_ids, user['liked_games'])
                diversity = self.calculate_diversity(recommendations[:12])
                
                ndcg_scores.append(ndcg)
                precision_scores.append(precision)
                recall_scores.append(recall)
                hit_rates.append(hit_rate)
                mrr_scores.append(mrr)
                diversity_scores.append(diversity)
                
            except Exception as e:
                logger.error(f"评估用户 {user['user_id']} 时出错: {e}")
                continue
        
        # 计算覆盖率
        coverage = self.calculate_coverage(all_recommendations)
        
        # 汇总结果
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'n_users_evaluated': len(ndcg_scores),
            'metrics': {
                'accuracy': {
                    'ndcg@12': {
                        'mean': float(np.mean(ndcg_scores)),
                        'std': float(np.std(ndcg_scores)),
                        'min': float(np.min(ndcg_scores)),
                        'max': float(np.max(ndcg_scores))
                    },
                    'precision@12': {
                        'mean': float(np.mean(precision_scores)),
                        'std': float(np.std(precision_scores))
                    },
                    'recall@12': {
                        'mean': float(np.mean(recall_scores)),
                        'std': float(np.std(recall_scores))
                    },
                    'hit_rate@12': {
                        'mean': float(np.mean(hit_rates)),
                        'std': float(np.std(hit_rates))
                    },
                    'mrr': {
                        'mean': float(np.mean(mrr_scores)),
                        'std': float(np.std(mrr_scores))
                    }
                },
                'diversity': {
                    'mean': float(np.mean(diversity_scores)),
                    'std': float(np.std(diversity_scores))
                },
                'coverage': float(coverage),
                'performance': {
                    'avg_recommendations_per_user': 20,
                    'total_unique_games_recommended': len(set(sum(all_recommendations, [])))
                }
            },
            'interpretation': self._interpret_results(np.mean(ndcg_scores), np.mean(hit_rates), 
                                                    np.mean(diversity_scores), coverage)
        }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _interpret_results(self, ndcg: float, hit_rate: float, diversity: float, coverage: float) -> Dict[str, str]:
        """解释评估结果"""
        interpretation = {}
        
        # NDCG解释
        if ndcg >= 0.8:
            interpretation['ndcg'] = "优秀 - 推荐排序质量非常高"
        elif ndcg >= 0.6:
            interpretation['ndcg'] = "良好 - 推荐排序质量较好"
        elif ndcg >= 0.4:
            interpretation['ndcg'] = "中等 - 推荐排序质量一般"
        else:
            interpretation['ndcg'] = "较差 - 推荐排序质量需要改进"
        
        # Hit Rate解释
        if hit_rate >= 0.8:
            interpretation['hit_rate'] = "优秀 - 大部分用户能找到喜欢的游戏"
        elif hit_rate >= 0.6:
            interpretation['hit_rate'] = "良好 - 多数用户能找到喜欢的游戏"
        elif hit_rate >= 0.4:
            interpretation['hit_rate'] = "中等 - 部分用户能找到喜欢的游戏"
        else:
            interpretation['hit_rate'] = "较差 - 推荐命中率需要提升"
        
        # 多样性解释
        if diversity >= 0.7:
            interpretation['diversity'] = "优秀 - 推荐结果非常多样化"
        elif diversity >= 0.5:
            interpretation['diversity'] = "良好 - 推荐结果较为多样化"
        elif diversity >= 0.3:
            interpretation['diversity'] = "中等 - 推荐结果多样性一般"
        else:
            interpretation['diversity'] = "较差 - 推荐结果过于相似"
        
        # 覆盖率解释
        if coverage >= 0.3:
            interpretation['coverage'] = "优秀 - 系统能推荐大量不同游戏"
        elif coverage >= 0.2:
            interpretation['coverage'] = "良好 - 系统推荐覆盖面较广"
        elif coverage >= 0.1:
            interpretation['coverage'] = "中等 - 系统推荐覆盖面一般"
        else:
            interpretation['coverage'] = "较差 - 系统推荐过于集中"
        
        return interpretation
    
    def save_evaluation_report(self, filepath: str = 'evaluation_report.json'):
        """保存评估报告"""
        if not self.evaluation_results:
            logger.error("没有评估结果可保存")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估报告已保存到: {filepath}")
    
    def print_evaluation_summary(self):
        """打印评估摘要"""
        if not self.evaluation_results:
            print("没有可用的评估结果")
            return
        
        metrics = self.evaluation_results['metrics']
        
        print("\n" + "="*60)
        print("桌游推荐系统评估报告")
        print("="*60)
        
        print(f"\n评估用户数: {self.evaluation_results['n_users_evaluated']}")
        print(f"评估时间: {self.evaluation_results['timestamp']}")
        
        print("\n📊 准确性指标:")
        print(f"  - NDCG@12: {metrics['accuracy']['ndcg@12']['mean']:.3f} (±{metrics['accuracy']['ndcg@12']['std']:.3f})")
        print(f"  - Precision@12: {metrics['accuracy']['precision@12']['mean']:.3f}")
        print(f"  - Recall@12: {metrics['accuracy']['recall@12']['mean']:.3f}")
        print(f"  - Hit Rate@12: {metrics['accuracy']['hit_rate@12']['mean']:.3f}")
        print(f"  - MRR: {metrics['accuracy']['mrr']['mean']:.3f}")
        
        print("\n🎯 多样性指标:")
        print(f"  - 推荐多样性: {metrics['diversity']['mean']:.3f}")
        print(f"  - 系统覆盖率: {metrics['coverage']:.3f}")
        print(f"  - 推荐游戏总数: {metrics['performance']['total_unique_games_recommended']}")
        
        print("\n📝 结果解释:")
        for metric, interpretation in self.evaluation_results['interpretation'].items():
            print(f"  - {metric}: {interpretation}")
        
        print("\n" + "="*60)


# 运行评估的主函数
def run_evaluation(data_path: str = 'data/BGG_Data.csv', 
                  user_ratings_path: str = None,
                  n_users: int = 100):
    """运行完整的系统评估"""
    
    print("开始加载推荐系统...")
    
    # 导入推荐系统
    from enhanced_recommendation import EnhancedRecommendationSystem
    
    # 创建推荐系统实例
    recommender = EnhancedRecommendationSystem()
    recommender.load_and_preprocess_data(data_path, user_ratings_path)
    
    print(f"数据加载完成: {len(recommender.df)} 个游戏")
    
    # 创建评估器
    evaluator = RecommendationEvaluator(recommender, recommender.df)
    
    # 运行评估
    print(f"\n开始评估 (创建 {n_users} 个合成用户)...")
    results = evaluator.evaluate_system(n_synthetic_users=n_users)
    
    # 打印结果
    evaluator.print_evaluation_summary()
    
    # 保存报告
    evaluator.save_evaluation_report('evaluation_report.json')
    
    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行评估
    results = run_evaluation(n_users=100)
