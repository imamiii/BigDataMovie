# data_loader.py
import pandas as pd
import numpy as np
import re
from config import EXCEL_FILE, COLUMN_NAMES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, excel_path=EXCEL_FILE):
        self.excel_path = excel_path
        self.df = None
        self.raw_columns = None

    def load_data(self):
        """加载Excel数据"""
        try:
            logger.info(f"正在加载数据文件: {self.excel_path}")
            self.df = pd.read_excel(self.excel_path)
            self.raw_columns = self.df.columns.tolist()
            logger.info(f"数据加载成功，共 {len(self.df)} 条记录，{len(self.df.columns)} 列")
            logger.info(f"原始列名: {self.raw_columns}")
            return self.df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def rename_columns(self, column_mapping=None):
        """重命名列"""
        if column_mapping is None:
            column_mapping = COLUMN_NAMES

        # 只重命名存在的列
        existing_mapping = {k: v for k, v in column_mapping.items()
                            if k in self.df.columns}
        self.df = self.df.rename(columns=existing_mapping)
        logger.info(f"列名已重命名: {existing_mapping}")
        return self.df

    def clean_data(self):
        """数据清洗"""
        # 去除空白行
        original_len = len(self.df)
        self.df = self.df.dropna(subset=['name', 'directors'])
        logger.info(f"去除空白记录: {original_len} -> {len(self.df)}")

        # 处理评分列
        if 'rating' in self.df.columns:
            # 新文件中的评分已经是数值类型，只需要处理缺失值
            self.df['rating'] = self.df['rating'].fillna(0.0)
            logger.info(f"评分列已清理，有效评分: {self.df['rating'].notna().sum()}")

        # 处理影人列（导演、演员）
        for col in ['directors', 'actors']:
            if col in self.df.columns:
                # 填充缺失值
                self.df[col] = self.df[col].fillna('未知')
                # 去除多余空格
                self.df[col] = self.df[col].astype(str).str.strip()
                logger.info(f"{col}列已清理")

        # 处理类型列
        if 'types' in self.df.columns:
            self.df['types'] = self.df['types'].fillna('未知')
            self.df['types'] = self.df['types'].astype(str).str.strip()
            logger.info(f"types列已清理")

        return self.df

    def extract_persons_from_string(self, person_str, separator='/'):
        """从字符串中提取影人列表"""
        if pd.isna(person_str) or person_str == '未知' or person_str == 'nan':
            return []

        persons = []
        for person in str(person_str).split(separator):
            person = person.strip()
            if person and person != '...' and len(person) > 1:
                persons.append(person)
        return persons

    def validate_data(self):
        """验证数据质量"""
        logger.info("=== 数据验证报告 ===")
        logger.info(f"总记录数: {len(self.df)}")

        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            logger.info(f"{col}: 缺失值 {missing} ({missing / len(self.df) * 100:.1f}%)")

        if 'rating' in self.df.columns:
            logger.info(f"评分统计: 最小值={self.df['rating'].min():.1f}, "
                        f"最大值={self.df['rating'].max():.1f}, "
                        f"平均值={self.df['rating'].mean():.1f}")

        return self.df

    def save_cleaned_data(self, output_path='data/cleaned_movies.csv'):
        """保存清洗后的数据"""
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"清洗后的数据已保存到: {output_path}")
        return output_path