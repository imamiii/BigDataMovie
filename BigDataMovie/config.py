# config.py
import os

# 项目路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXCEL_FILE = os.path.join(DATA_DIR, 'ChineseMoviesData.csv')

# 列名映射（根据新的ChineseMoviesData.csv调整）
COLUMN_NAMES = {
    'Name': 'name',  # 电影名称
    'Director': 'directors',  # 导演
    'Actor': 'actors',  # 演员
    'Types': 'types',  # 类型
    'Score': 'rating'  # 评分
}

# 网络分析参数
RATING_THRESHOLD = 6.0  # 低分电影阈值
MIN_COOPERATION_COUNT = 2  # 最小合作次数
COMMUNITY_DETECTION_METHOD = 'louvain'  # 社区检测方法: louvain, girvan_newman, label_propagation

# 可视化参数
NODE_SIZE_MULTIPLIER = 50
EDGE_WIDTH_MULTIPLIER = 2
MAX_NODES_TO_LABEL = 50