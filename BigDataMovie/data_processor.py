# data_processor.py
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.persons_dict = defaultdict(list)  # 影人 -> 参与的电影列表
        self.cooperation_counts = Counter()    # 影人合作次数统计

    def extract_all_persons(self):
        """提取所有影人及其参与的电影（支持同人多角色）"""
        # 用于去重： (person, movie_id, role)
        seen = set()

        for idx, row in self.df.iterrows():
            movie_id = row.get('id', idx)
            movie_name = row.get('name', f'电影_{idx}')
            rating = row.get('rating', np.nan)

            directors = self._extract_persons(row.get('directors', ''))
            for director in directors:
                key = (director, movie_id, 'director')
                if key not in seen:
                    seen.add(key)
                    self.persons_dict[director].append({
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'role': 'director',
                        'rating': rating
                    })

            writers = self._extract_persons(row.get('writers', ''))
            for writer in writers:
                key = (writer, movie_id, 'writer')
                if key not in seen:
                    seen.add(key)
                    self.persons_dict[writer].append({
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'role': 'writer',
                        'rating': rating
                    })

            actors = self._extract_persons(row.get('actors', ''))
            for actor in actors:
                key = (actor, movie_id, 'actor')
                if key not in seen:
                    seen.add(key)
                    self.persons_dict[actor].append({
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'role': 'actor',
                        'rating': rating
                    })

        logger.info(f"共提取 {len(self.persons_dict)} 位影人")
        return self.persons_dict

    def _extract_persons(self, person_str):
        if pd.isna(person_str):
            return []
        persons = []
        for person in str(person_str).split('/'):
            person = person.strip()
            if person and person != '...' and len(person) > 1:
                persons.append(person)
        return persons

    def _extract_types(self, types_str):
        if pd.isna(types_str):
            return []
        s = str(types_str).strip()
        if not s or s in ("未知", "nan"):
            return []
        for sep in ["、", ",", "，", "|"]:
            s = s.replace(sep, "/")
        parts = [p.strip() for p in s.split("/") if p.strip()]
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def build_cooperation_edges(self):
        """构建合作边：导演-导演，导演-演员（保持你现有功能不变）"""
        edges = []

        for idx, row in self.df.iterrows():
            movie_id = row.get('id', idx)
            movie_name = row.get('name', f'电影_{idx}')
            rating = row.get('rating', np.nan)
            types_list = self._extract_types(row.get('types', ''))

            directors = self._extract_persons(row.get('directors', ''))
            actors = self._extract_persons(row.get('actors', ''))

            # 1) 导演-导演
            for i in range(len(directors)):
                for j in range(i + 1, len(directors)):
                    d1, d2 = directors[i], directors[j]
                    edges.append({
                        'person1': d1,
                        'person2': d2,
                        'role1': 'director',
                        'role2': 'director',
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'rating': rating,
                        'types': types_list,
                        'year': row.get('year', 0),
                        'cooperation_id': f"{d1}_{d2}_{movie_id}"
                    })
                    self.cooperation_counts[tuple(sorted([d1, d2]))] += 1

            # 2) 导演-演员
            for d in directors:
                for a in actors:
                    edges.append({
                        'person1': d,
                        'person2': a,
                        'role1': 'director',
                        'role2': 'actor',
                        'movie_id': movie_id,
                        'movie_name': movie_name,
                        'rating': rating,
                        'types': types_list,
                        'year': row.get('year', 0),
                        'cooperation_id': f"{d}_{a}_{movie_id}"
                    })
                    self.cooperation_counts[tuple(sorted([d, a]))] += 1

        edges_df = pd.DataFrame(edges)
        logger.info(f"构建了 {len(edges_df)} 条合作关系边")
        logger.info(f"涉及 {len(self.cooperation_counts)} 对不同的影人组合")
        return edges_df

    def get_person_stats(self):
        person_stats = []

        for person, movies in self.persons_dict.items():
            ratings = [m['rating'] for m in movies if not pd.isna(m['rating'])]
            avg_rating = np.mean(ratings) if ratings else np.nan

            roles = [m['role'] for m in movies]
            role_counts = Counter(roles)

            stat = {
                'name': person,
                'movie_count': len(movies),  # 维持原逻辑：按“参与记录数”计（导演/演员都算）
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
        movie_stats = []

        for idx, row in self.df.iterrows():
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