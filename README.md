# cv_baseline

Создание виртуальной среды
```cmd
conda create --name <env> --file requirements.txt
pip install -r requirements_pip.txt
```

Скачать и переместить файл с данными subset.zip в директорию data/raw

Запустить пайплайн подготовки данных и обучения модели

```cmd
dvc repro
```

Файл с лучшей моделью
* ./outs/model/best_model.ckpt

Изменение лоса/метрик mlflow 
* ./outs

Файл с архитектурой модели
- src/model.py

Файл с подсчетом метрики DICE Coef 
- src/metric.py

Файл с циклом обучения
- src/train.py

Файл с функцией предобработки данных 
- src/data_preprocess.py

График функции ошибки
![BCELoss](https://github.com/chmelevme/cv_baseline/blob/main/images/Loss.PNG)

График метрики DICE Coef 
![Dice coef](https://github.com/chmelevme/cv_baseline/blob/main/images/Dice.PNG)
