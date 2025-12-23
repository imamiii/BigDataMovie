# data_processor.py
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import logging
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.persons_dict = defaultdict(list)  # 影人 -> 参与的电影列表
        self.cooperation_counts = Counter()  # 影人合作次数统计

    def extract_all_persons(self):
        """提取所有影人及其参与的电影"""
        for idx, row in self.df.iterrows():
            movie_id = row.get('id', idx)
            movie_name = row.get('name', f'电影_{idx}')
            rating = row.get('rating', np.nan)

            # 提取导演
            directors = self._extract_persons(row.get('directors', ''))
            for director in directors:
                self.persons_dict[director].append({
                    'movie_id': movie_id,
                    'movie_name': movie_name,
                    'role': 'director',
                    'rating': rating
                })

            # 提取编剧
            writers = self._extract_persons(row.get('writers', ''))
            for writer in writers:
                self.persons_dict[writer].append({
                    'movie_id': movie_id,
                    'movie_name': movie_name,
                    'role': 'writer',
                    'rating': rating
                })

            # 提取演员
            actors = self._extract_persons(row.get('actors', ''))
            for actor in actors:
                self.persons_dict[actor].append({
                    'movie_id': movie_id,
                    'movie_name': movie_name,
                    'role': 'actor',
                    'rating': rating
                })

        logger.info(f"共提取 {len(self.persons_dict)} 位影人")
        return self.persons_dict

    def _extract_persons(self, person_str):
        """提取影人列表"""
        if pd.isna(person_str):
            return []

        persons = []
        for person in str(person_str).split('/'):
            person = person.strip()
            if person and person != '...' and len(person) > 1:
                # 保留完整姓名，不再截断中间有点(·)的名字
                persons.append(person)
        return persons

    def _extract_types(self, types_str):
        """把类型字段统一解析为 list[str]，兼容 / 、 , | 等分隔符"""
        if pd.isna(types_str):
            return []
        s = str(types_str).strip()
        if not s or s in ("未知", "nan"):
            return []
        # 兼容多种分隔符
        for sep in ["、", ",", "，", "|"]:
            s = s.replace(sep, "/")
        parts = [p.strip() for p in s.split("/") if p.strip()]
        # 去重但保序
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def build_cooperation_edges(self):
        """构建影人合作关系边列表 - 计算导演和演员之间的一对一关系，以及导演之间的合作"""
        edges = []

        for idx, row in self.df.iterrows():
            movie_id = row.get('id', idx)
            movie_name = row.get('name', f'电影_{idx}')
            rating = row.get('rating', np.nan)
            types_list = self._extract_types(row.get('types', ''))

            # 只提取导演和演员
            directors = self._extract_persons(row.get('directors', ''))
            actors = self._extract_persons(row.get('actors', ''))

            # 1. 导演之间的合作
            for i in range(len(directors)):
                for j in range(i + 1, len(directors)):
                    director1 = directors[i]
                    director2 = directors[j]
                    # 创建边
                    edge = {
                        'person1': director1,
                        'person2': director2,
                        'role1': 'director',
                        'role2': 'director',
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'rating': rating,
                        'types': types_list,
                        'year': row.get('year', 0),
                        'cooperation_id': f"{director1}_{director2}_{movie_id}"
                    }
                    edges.append(edge)

                    # 统计合作次数
                    pair_key = tuple(sorted([director1, director2]))
                    self.cooperation_counts[pair_key] += 1

            # 2. 导演和演员之间的一对一关系
            for director in directors:
                for actor in actors:
                    # 创建边
                    edge = {
                        'person1': director,
                        'person2': actor,
                        'role1': 'director',
                        'role2': 'actor',
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'rating': rating,
                        'types': types_list,
                        'year': row.get('year', 0),
                        'cooperation_id': f"{director}_{actor}_{movie_id}"
                    }
                    edges.append(edge)

                    # 统计合作次数
                    pair_key = tuple(sorted([director, actor]))
                    self.cooperation_counts[pair_key] += 1

        edges_df = pd.DataFrame(edges)
        logger.info(f"构建了 {len(edges_df)} 条合作关系边")
        logger.info(f"涉及 {len(self.cooperation_counts)} 对不同的影人组合")

        return edges_df

    def get_person_stats(self):
        """获取影人统计信息"""
        person_stats = []

        for person, movies in self.persons_dict.items():
            ratings = [m['rating'] for m in movies if not pd.isna(m['rating'])]
            avg_rating = np.mean(ratings) if ratings else np.nan

            # 按角色统计
            roles = [m['role'] for m in movies]
            role_counts = Counter(roles)

            stat = {
                'name': person,
                'movie_count': len(movies),
                'avg_rating': avg_rating,
                'director_count': role_counts.get('director', 0),
                'writer_count': role_counts.get('writer', 0),
                'actor_count': role_counts.get('actor', 0)
            }
            person_stats.append(stat)

        stats_df = pd.DataFrame(person_stats)
        logger.info(f"影人统计信息已生成，共 {len(stats_df)} 位影人")
        return stats_df

    def get_movie_stats(self):
        """获取电影统计信息"""
        movie_stats = []

        for idx, row in self.df.iterrows():
            # 计算每部电影的影人数量
            directors_count = len(self._extract_persons(row.get('directors', '')))
            writers_count = len(self._extract_persons(row.get('writers', '')))
            actors_count = len(self._extract_persons(row.get('actors', '')))
            total_persons = directors_count + writers_count + actors_count

            stat = {
                'movie_id': row.get('id', idx),
                'name': row.get('name', ''),
                'year': row.get('year', 0),
                'rating': row.get('rating', np.nan),
                'directors_count': directors_count,
                'writers_count': writers_count,
                'actors_count': actors_count,
                'total_persons': total_persons
            }
            movie_stats.append(stat)

        stats_df = pd.DataFrame(movie_stats)
        logger.info(f"电影统计信息已生成，共 {len(stats_df)} 部电影")
        return stats_df