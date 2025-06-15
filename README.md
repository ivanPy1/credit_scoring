# Loan Approval Prediction (Full Pipeline ML Project)

Полный проект машинного обучения для предсказания одобрения кредита. Поддерживает два режима:

- RandomForestClassifier (CPU) 

- XGBoostClassifier (GPU, CUDA-ускорение)

Включает в себя подготовку данных, построение признаков, подбор гиперпараметров, объяснение модели с SHAP, а также визуализацию метрик и выводов.

## Описание

- Используются числовые и категориальные признаки.
- Производится обработка пропусков, масштабирование числовых признаков и кодирование категориальных.
- Был проведен масштабный подбор гиперпараметров для каждой модели через GridSearchCV и RandomizedSearchCV.
- Добавляется новый признак `AssetHealth = NetWorth / (TotalAssets + 1)`.
- Удаляются избыточные признаки: `NetWorth`, `TotalAssets`, `AnnualIncome`.
- Результаты оцениваются с помощью отчёта классификации.
- Поддержка XGBoost с device='cuda' для ускорения на GPU.
- SHAP-графики важности признаков и ROC-кривые.

## Структура проекта

```
Loan_final/
├── config.py               # Глобальные параметры и пути
├── data_loader.py          # Загрузка и разбиение данных
├── features.py             # Обработка признаков (включая синтетические)
├── model_rf.py             # Модель на RandomForest (CPU)
├── model_xgb.py            # Модель на XGBoost (GPU)
├── evaluate_rf.py          # Оценка RandomForest модели
├── evaluate_xgb.py         # Оценка XGBoost модели
├── shapExp_rf.py           # SHAP-анализ для RandomForest
├── shapExp_xgb.py          # SHAP-анализ для XGBoost
├── main_rf.py              # Точка входа для RF
├── main_xgb.py             # Точка входа для XGB
├── requirements.txt        # Зависимости
└── README.md               # Документация проекта
```

## Установка

1. Скопируйте проект в рабочую директорию.
2. Скачайте файл `Loan.csv` с Kaggle (https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv). 
3. Поместите файл `Loan.csv` в корень проекта.
4. Установите зависимости:

```
pip install -r requirements.txt
```

## Запуск моделей

1. RandomForest (CPU)
```
python main_rf.py
```

2. XGBoost (GPU)
```
python main_xgb.py
```

## Метрики

При запуске модели через main вы получите:

- classification report: accuracy, precision, recall, f1

- ROC AUC

## SHAP-анализ

Для запуска визуального объяснения модели:

```
python shapExp_rf.py     # Для RandomForest
```
```
python shapExp_xgb.py    # Для XGBoost
```

Это создаст:

- SHAP summary график (влияние признаков)

- ROC кривая

## Результат

После запуска будет выведен отчёт классификации с основными метриками: точность, полнота, F1-мера.

## Примечания

- Для XGBoost необходима поддержка CUDA!!!
- При расчёте `AssetHealth` используется деление на `TotalAssets + 1` для избежания деления на ноль.