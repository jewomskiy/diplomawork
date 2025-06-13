import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import set_random_seed
from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root
from sklearn.metrics import mean_absolute_error, r2_score


def run_regression_example(visualise: bool = False, with_tuning: bool = True,
                           timeout: float = 10., preset: str = 'auto', use_stats: bool = False):
    data_path = f'prepared_data.csv'

    print(data_path)

    data = InputData.from_csv(data_path, target_columns='Arr_Delay',
                              task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)
    problem = 'regression'

    composer_params = {'history_dir': 'custom_history_dir', 'preset': preset}
    auto_model = Fedot(problem=problem, seed=42, timeout=timeout, logging_level=logging.FATAL,
                       with_tuning=with_tuning, **composer_params)

    auto_model.fit(features=train)
    prediction = auto_model.predict(features=test)
    print(prediction)

    if visualise:
        auto_model.history.save('saved_regression_history.json')
        auto_model.plot_prediction()

    print(auto_model.get_metrics())
    y_test = test.target
    predicted = prediction

    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    print(f'MAE: {mae:.2f}, R²: {r2:.2f}')
    return prediction



if __name__ == '__main__':
    set_random_seed(42)

    # data = pd.read_csv('data/prepared_data.csv')
    #
    # # Разделение данных (80% train / 20% test)
    # train_data, test_data = train_test_split(
    #     data,
    #     test_size=0.2,
    #     random_state=42,  # для воспроизводимости
    #     stratify=data['Arr_Delay']  # стратификация по целевой переменной
    # )
    #
    # # Сохранение результатов
    # train_data.to_csv('train_data.csv', index=False)
    # test_data.to_csv('test_data.csv', index=False)
    #
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # full_path_train = 'train_data.csv'
    # full_path_test = 'test_data.csv'
    run_regression_example(visualise=True)