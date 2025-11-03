import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, precision_score,
    recall_score, roc_curve, accuracy_score, classification_report
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder

import plots as p

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder


def EDA_preprocess(df, seed=42, test_size=0.2):
    """
    - Очистка данных (удаление ненужных колонок и строк с пропусками в таргете)
    - Преобразование даты
    - Заполнение пропусков
    - Разделение на train/test
    - Возврат подготовленных данных
    """

    df = df.copy()

    # === 1. Первичная очистка ===
    drop_cols = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Удаляем строки без целевой переменной
    df = df.dropna(subset=['RainTomorrow'])

    # === 2. Обработка даты ===
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)

    # === 3. Формирование X и y ===
    y = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    X = df.drop(columns=['RainTomorrow'])

    # === 4. Обработка категориальных признаков ===
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna('missing').astype(str)

    if 'RainToday' in X.columns:
        X['RainToday'] = X['RainToday'].map({'No': 0, 'Yes': 1}).fillna(-1)

    # === 5. Разделение на train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=test_size, stratify=y
    )

    # === 6. Определение типов признаков ===
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # === 7. Пайплайн предобработки ===
    num_transformer = SimpleImputer(strategy='mean')
    cat_transformer = TargetEncoder(handle_unknown='use_encoded_value', handle_missing='value')

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='passthrough')

    preprocessor_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # === 8. Трансформация ===
    X_train_transformed = preprocessor_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor_pipeline.transform(X_test)

    # === 9. Преобразуем обратно в DataFrame ===
    transformed_columns = num_cols + cat_cols + [
        c for c in X_train.columns if c not in num_cols + cat_cols
    ]

    X_train_preprocessed = pd.DataFrame(X_train_transformed, index=X_train.index, columns=transformed_columns)
    X_test_preprocessed = pd.DataFrame(X_test_transformed, index=X_test.index, columns=transformed_columns)

    # === 10. Возврат всех данных ===
    return X, y, X_train, y_train, X_test, y_test, X_train_preprocessed, X_test_preprocessed


def evaluate_models_cv(models, X, y, cv=5, seed=None, use_smote=False):
    """
    Проводит кросс-валидацию для набора моделей с корректной обработкой признаков и опцией SMOTE.
    
    Параметры:
        models (list): список кортежей ('ModelName', model_object)
        X, y: признаки и целевая переменная
        cv (int или объект KFold): число фолдов или объект кросс-валидации
        seed (int): случайное зерно для воспроизводимости
        use_smote (bool): если True — добавляет SMOTE внутри Pipeline

    Возвращает:
        pd.DataFrame с усреднёнными метриками
    """

    # корректно создаём StratifiedKFold
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        cv_splitter = cv

    # определяем числовые и категориальные признаки
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # трансформеры
    num_transformer = SimpleImputer(strategy='mean')
    cat_transformer = TargetEncoder(handle_unknown='value', handle_missing='value')

    # единый препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop'  # отбрасываем ненужные колонки, если они есть
    )

    # набор метрик
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc'
    }

    all_metrics = {}

    # цикл по моделям
    for name, model in models:
        print(f"\n{'='*60}\nКросс-валидация модели: {name}\n{'='*60}")

        # фиксируем seed
        if seed is not None and hasattr(model, 'random_state'):
            model.set_params(random_state=seed)

        # собираем пайплайн
        steps = [('preprocessor', preprocessor)]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=seed)))
        steps.append(('model', model))
        pipeline = ImbPipeline(steps)

        # кросс-валидация
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # усредняем метрики
        metrics_mean = {metric.replace("test_", ""): np.mean(values)
                        for metric, values in cv_results.items()
                        if metric.startswith("test_")}

        all_metrics[name] = metrics_mean

        # краткий вывод
        print(f"Accuracy:  {metrics_mean['accuracy']:.4f}")
        print(f"Precision: {metrics_mean['precision']:.4f}")
        print(f"Recall:    {metrics_mean['recall']:.4f}")
        print(f"F1 Score:  {metrics_mean['f1']:.4f}")
        print(f"ROC AUC:   {metrics_mean['roc_auc']:.4f}")

    # объединяем результаты в таблицу
    df_results = pd.DataFrame(all_metrics).T.sort_values(by="roc_auc", ascending=False)
    print("\n=== Сводная таблица метрик (усреднённые по CV) ===")
    print(df_results.to_string(float_format="%.4f"))

    return df_results

def compare_datasets_results(df1, df2, name1="Dataset_1", name2="Dataset_2", plot=True):
    """
    Сравнивает результаты моделей между двумя наборами данных.

    Параметры:
        df1, df2: DataFrame — таблицы метрик (из evaluate_models_cv)
        name1, name2: подписи для наборов данных
        plot: bool — строить ли графики

    Возвращает:
        DataFrame с разницей метрик (df2 - df1)
    """
    # Проверим, что модели совпадают
    common_models = df1.index.intersection(df2.index)
    if len(common_models) == 0:
        raise ValueError("Нет общих моделей для сравнения!")

    # Вычисляем разницу (вторая - первая)
    diff = df2.loc[common_models] - df1.loc[common_models]
    diff = diff.round(4)

    print(f"\n=== Разница метрик ({name2} - {name1}) ===")
    print(diff)

    # Визуализация
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(diff.columns), figsize=(18, 4), sharey=True)
        fig.suptitle(f"Изменение метрик между {name1} и {name2}", fontsize=14)

        for ax, metric in zip(axes, diff.columns):
            diff[metric].plot(kind='bar', ax=ax, title=metric)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return diff
def make_preprocessor(X, num_cols, cat_cols):
    available_cols = X.columns.tolist()
    num_cols = [c for c in num_cols if c in available_cols]
    cat_cols = [c for c in cat_cols if c in available_cols]
    
    print("Используемые числовые признаки:", num_cols)
    print("Используемые категориальные признаки:", cat_cols)
    
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='mean'), num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough')
def evaluate_classification(y_test, y_pred, y_probs=None, model_name="Model"):
    print(f"=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label='Yes'))
    if y_probs is not None:
        y_test_num = y_test.map({'No':0, 'Yes':1})
        print("ROC AUC:", roc_auc_score(y_test_num, y_probs))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_metrics(y_test, y_pred, y_probs=None, model_name="Model"):
    """
    Выводит основные метрики классификации и возвращает их в виде словаря.
    Поддерживает как бинарные метки (0/1), так и строковые ('Yes'/'No').
    """
    print(f"=== {model_name} ===")
    
    # Преобразуем метки в числа для ROC AUC, если это 'Yes'/'No'
    if hasattr(y_test, "map") and set(y_test.unique()) == {'Yes', 'No'}:
        y_test_num = y_test.map({'No': 0, 'Yes': 1})
        pos_label = 'Yes'
    else:
        y_test_num = y_test
        pos_label = 1  # для числовых меток

    # Вычисляем метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=pos_label)
    rec = recall_score(y_test, y_pred, pos_label=pos_label)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    roc_auc = roc_auc_score(y_test_num, y_probs) if y_probs is not None else None

    # Выводим на экран
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC:   {roc_auc:.4f}")

    # Возвращаем словарь
    metrics_dict = {
        f"{model_name}_Accuracy": acc,
        f"{model_name}_Precision": prec,
        f"{model_name}_Recall": rec,
        f"{model_name}_F1": f1,
    }
    if roc_auc is not None:
        metrics_dict[f"{model_name}_ROC_AUC"] = roc_auc
    
    return metrics_dict


def compare_model_metrics(metrics1: dict, metrics2: dict, name1: str = "Model_1", name2: str = "Model_2"):
    """
    Сравнивает метрики двух моделей и выводит разницу.

    Параметры:
        metrics1 (dict): метрики первой модели
        metrics2 (dict): метрики второй модели
        name1 (str): имя первой модели (для таблицы)
        name2 (str): имя второй модели (для таблицы)

    Возвращает:
        pd.DataFrame — таблица с метриками и их разницей
    """
    # Извлекаем все метрики, которые есть хотя бы в одном словаре
    all_metrics = set(metrics1.keys()).union(metrics2.keys())
    
    rows = []
    for metric in all_metrics:
        val1 = metrics1.get(metric, None)
        val2 = metrics2.get(metric, None)
        diff = None if (val1 is None or val2 is None) else val2 - val1
        rows.append({
            "Metric": metric.replace(f"{name1}_", "").replace(f"{name2}_", ""),
            f"{name1}": val1,
            f"{name2}": val2,
            "Difference": diff
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Metric").reset_index(drop=True)

    # Красивый вывод
    print(f"\n=== Сравнение метрик: {name1} vs {name2} ===")
    print(df.to_string(index=False, float_format="%.4f"))

    return df

def one_hot_encode_df(df, categorical_cols, drop_original=True)->pd.DataFrame:
    """
    Делает One-Hot кодирование выбранных категориальных колонок.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный DataFrame.
    categorical_cols : list
        Список названий категориальных колонок для кодирования.
    drop_original : bool, default=True
        Если True — удаляет исходные категориальные колонки.
    
    Возвращает:
    -----------
    pd.DataFrame — копию исходного df с добавленными one-hot признаками.
    """
    
    # Копия исходного DataFrame
    df_encoded = df.copy()
    
    # Инициализация OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    
    # Преобразование категориальных данных
    encoded = encoder.fit_transform(df_encoded[categorical_cols])
    
    # Получаем имена новых колонок
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    
    # Создаём DataFrame с one-hot признаками
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df_encoded.index)
    
    # Добавляем новые колонки
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    # Удаляем исходные, если нужно
    if drop_original:
        df_encoded = df_encoded.drop(columns=categorical_cols)
    
    return df_encoded
def target_encoded_df(df, categorical_col, target, drop_original=True)->pd.DataFrame:
    df_encoded = df.copy()
    df_encoded[target] = df_encoded[target].map({'Yes': 1, 'No': 0})
    for col in categorical_col:
        mean_target = df_encoded.groupby(col)[target].mean()
        df_encoded[col + '_encoded'] = df_encoded[col].map(mean_target)
    if drop_original:
        df_encoded = df_encoded.drop(columns=categorical_col)
    return df_encoded
def data_encoded_df(df, data_col, drop_original=True)->pd.DataFrame:
    df_encoded=df.copy()
    df_encoded[data_col]=pd.to_datetime(df_encoded[data_col], format='%Y-%m-%d')
    df_encoded['Year'] = df_encoded[data_col].dt.year
    df_encoded['Month'] = df_encoded[data_col].dt.month
    df_encoded['Day'] = df_encoded[data_col].dt.day
    if drop_original:
        df_encoded = df_encoded.drop(columns=data_col)
    return df_encoded
    
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'