import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. 读取数据 ---
train_df = pd.read_csv(r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/train.csv")
test_df = pd.read_csv(r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/test.csv")

# --- 2. 准备特征 ---
X = train_df.drop(['id', 'exam_score'], axis=1)
y = train_df['exam_score']
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# --- 3. 预处理管道 (沿用之前的配置) ---
# --- 2.1 检查缺失值 ---
print("训练集缺失值情况:\n", train_df.isnull().sum())

# 如果有缺失值，我们可以用“中位数”填充数值，用“众数”填充类别
# (这里写一个简单的自动填充逻辑，防止报错)
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
    else:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].median())

# --- 2.2 去除重复值 ---
train_df.drop_duplicates(inplace=True)

# --- 2.3 分离特征(X)和目标(y) ---
# ID 列对预测分数没有帮助，必须去掉，但测试集的ID要留着最后提交用
X = train_df.drop(['id', 'exam_score'], axis=1)
y = train_df['exam_score']

# 测试集我们也先把ID拿出来保存，剩下的作为特征
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# 区分数值列和类别列
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
categorical_features = ['gender', 'course', 'internet_access', 'study_method', 
                        'sleep_quality', 'facility_rating', 'exam_difficulty']

# 定义数值型数据的处理步骤：补充缺失值 -> 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 定义类别型数据的处理步骤：补充缺失值 -> 独热编码(OneHot)
# handle_unknown='ignore' 很重要，防止测试集中出现训练集没见过的类别导致报错
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合起来
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("预处理管道构建完成！")
# ... (此处省略之前定义的 preprocessor 代码) ...

# --- 4. GPU 模型训练 ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        tree_method='hist',
        device='cuda', 
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6
    ))
])

print("开始 GPU 加速训练...")
model.fit(X, y) # 使用全部数据进行最终训练

# --- 5. 预测与保存 ---
print("正在预测测试集...")
final_preds = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_preds
})

submission.to_csv("D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/submission_gpu.csv", index=False)
print("文件已成功保存：submission_gpu.csv")