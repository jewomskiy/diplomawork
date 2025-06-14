import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error
import numpy as np
import torch
# import gdown
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# from google.colab import files
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task
import seaborn as sns
from datetime import datetime


class PreparingCSV:
    def __init__(self, input_file, weather_file, output_file, ):
        self.input_file = input_file
        self.weather_file = weather_file
        self.output_file = output_file

    def load_data(self):
        print("Загрузка данных...")
        flights_data = pd.read_csv(self.input_file)
        weather_data = pd.read_csv(self.weather_file)
        print(flights_data.describe().to_csv("my_description.csv"))
        print(weather_data.describe().to_csv("my_description_weather.csv"))
        return flights_data, weather_data

    def preprocess(self, flights_data, weather_data, random_state):
        print("Предобработка данных...")

        flights_data.loc[:, 'Delay_LastAircraft'] = flights_data['Delay_LastAircraft'].apply(lambda x: max(x, 0))

        flights_data = flights_data.sample(frac=0.4, random_state=random_state).reset_index(drop=True)

        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])

        def clip_values(df, column, lower, upper):
            df[column] = np.clip(df[column], lower, upper)
            return df

        time_order = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
        flights_data['DepTime_order'] = flights_data['DepTime_label'].map(time_order)

        flights_data = flights_data.sort_values(
            by=['Tail_Number', 'FlightDate', 'DepTime_order']
        ).reset_index(drop=True)

        flights_data['Delay_LastAircraft'] = flights_data.groupby('Tail_Number')['Arr_Delay'].shift(1)
        flights_data['Delay_LastAircraft'] = flights_data['Delay_LastAircraft'].fillna(0)

        flights_data = clip_values(flights_data, 'Arr_Delay', 0, 60)

        flights_data = flights_data.sort_values(by='FlightDate').reset_index(drop=True)

        merged_data = pd.merge(
            flights_data,
            weather_data,
            how='left',
            left_on=['Arr_Airport', 'FlightDate'],
            right_on=['airport_id', 'FlightDate']
        )
        merged_data = pd.merge(
            merged_data,
            weather_data,
            how='left',
            left_on=['Dep_Airport', 'FlightDate'],
            right_on=['airport_id', 'FlightDate'],
            suffixes=('', '_dep')
        )

        merged_data.drop(columns=[
            'Dep_Delay_Tag', "Dep_Delay", "Dep_Delay_Type", "Arr_Delay_Type", "STATE",
            "Dep_CityName", "AIRPORT", "CITY", "Arr_CityName", "LATITUDE", "LONGITUDE", "Delay_NAS", "Delay_Security", "Delay_Weather", "Delay_Carrier"
        ], inplace=True, errors='ignore')

        merged_data.fillna(merged_data.median(numeric_only=True), inplace=True)

        def calculate_flight_order(data):
            time_order = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
            data['DepTime_order'] = data['DepTime_label'].map(time_order)

            data = data.sort_values(by=['Tail_Number', 'FlightDate', 'DepTime_order']).reset_index(drop=True)

            data['Flight_Order'] = data.groupby(['Tail_Number', 'FlightDate']).cumcount() + 1

            data['PreviousFlights_Delay'] = data.groupby(['Tail_Number', 'FlightDate'])['Delay_LastAircraft'].cumsum() - data['Delay_LastAircraft']
            data['PreviousFlights_Delay'].fillna(0, inplace=True)

            data['Daily_Departure_Count'] = data.groupby(['FlightDate', 'Dep_Airport'])['Flight_Order'].transform('count')
            data['DayOfWeek'] = data['FlightDate'].dt.dayofweek
            data['Month'] = data['FlightDate'].dt.month
            data['TempDiff'] = data['tmax'] - data['tmin']
            return data

        merged_data = merged_data[merged_data['Dep_Airport'].isin(['ATL'])]
        merged_data = calculate_flight_order(merged_data)

        merged_data = pd.get_dummies(merged_data, columns=['Airline', 'DepTime_label', 'Arr_Airport'])

        merged_data = merged_data.sort_values(
            by=['FlightDate', 'DepTime_order']
        ).reset_index(drop=True)

        numeric_data = merged_data[merged_data.columns[:16]].select_dtypes(include=['number'])


        if not numeric_data.empty and len(numeric_data.columns) > 1:
            plt.figure(figsize=(15, 15))
            correlation_matrix = numeric_data.corr()


            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt=".2f",
                linewidths=0.2
            )
            plt.title('Матрица корреляций признаков', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()


            plt.savefig('correlation_heatmap.png', dpi=300)
            plt.close()
            print("Тепловая карта корреляций сохранена как correlation_heatmap.png")
        else:
            print("Недостаточно числовых данных для построения тепловой карты")

        return merged_data

    def save_data(self, merged_data):

        merged_data.to_csv(self.output_file, index=False)
        print(f"Данные сохранены в {self.output_file}")
        print("Пример данных:")
        print(merged_data.head())


class TrainingModel:
    def __init__(self, data_path, target_name, n_threads, timeout, n_folds):
        self.data_path = data_path
        self.target_name = target_name
        self.n_threads = n_threads
        self.timeout = timeout
        self.n_folds = n_folds

    def load_and_split_data(self, random_state):
        data = pd.read_csv(self.data_path)
        data = data.dropna()

        train_data, test_data = train_test_split(
            data,
            test_size=0.3,
            random_state=random_state,
            shuffle=True
        )
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def train_model(self, train_data, test_data, random_state):
        task = Task('reg')

        roles = {'target': self.target_name}

        automl_model = TabularUtilizedAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            reader_params={
                'n_jobs': self.n_threads,
                'cv': self.n_folds,
                'random_state': random_state
            },
        )

        oof_predictions = automl_model.fit_predict(train_data, roles=roles, verbose=2)
        test_predictions = automl_model.predict(test_data)
        return automl_model, oof_predictions, test_predictions

def main():
    TARGET_NAME = 'Arr_Delay'
    RANDOM_STATE = int(input('Введите значение random state: '))
    N_THREADS = int(input('Введите число потоков: '))
    MIN = int(input('Введите время работы (в минутах): '))
    timeout = 60 * MIN

    with open('output.log', 'w') as log_file:
        sys.stdout = log_file
        try:
            np.random.seed(RANDOM_STATE)
            torch.set_num_threads(N_THREADS)

            flights_data = Path("US_flights_2023.csv")
            weather_data = Path("weather_meteo_by_airport.csv")
            if not flights_data.exists() or not weather_data.exists():
                gdown.download_folder('https://drive.google.com/drive/folders/18houVS5ebR_Bw3_lQ3bNZ5X09lhaFsim')

            preparator = PreparingCSV(
                input_file='US_flights_2023.csv',
                weather_file='weather_meteo_by_airport.csv',
                output_file='prepared_data.csv',
            )
            flights, weather = preparator.load_data()
            prepared_data = preparator.preprocess(flights, weather, RANDOM_STATE)
            preparator.save_data(prepared_data)

            model_trainer = TrainingModel(
                data_path='prepared_data.csv',
                target_name=TARGET_NAME,
                n_threads=N_THREADS,
                timeout=timeout,
                n_folds=10
            )
            train_data, test_data = model_trainer.load_and_split_data(RANDOM_STATE)
            automl_model, oof_predictions, test_predictions = model_trainer.train_model(train_data, test_data, RANDOM_STATE)

            accurate_fi = automl_model.get_feature_scores('accurate', test_data, silent=True).head(5)
            accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize=(30, 15), grid=True)

            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.savefig('feature_importance.png')
            plt.show()

            if np.any(np.isnan(test_predictions.data)) or np.any(np.isnan(test_data[TARGET_NAME].values)):
                print("Обнаружены NaN в предсказаниях или тестовых данных.")
            else:
                mse = mean_squared_error(test_data[TARGET_NAME].values, test_predictions.data)
                mae = mean_absolute_error(test_data[TARGET_NAME].values, test_predictions.data)
                r2 = r2_score(test_data[TARGET_NAME].values, test_predictions.data)
                print(f'MSE: {mse}', f'MAE: {mae}', f'R2: {r2}', sep='\n')

                oof_predictions.data[:, 0] = np.clip(oof_predictions.data[:, 0], 0, None)
                test_predictions.data[:, 0] = np.clip(test_predictions.data[:, 0], 0, None)

                oof_pred_exp = np.nan_to_num(np.exp(oof_predictions.data[:, 0]) - 1, nan=0.0)
                test_pred_exp = np.nan_to_num(np.exp(test_predictions.data[:, 0]) - 1, nan=0.0)
                oof_score = root_mean_squared_log_error(train_data[TARGET_NAME].values, oof_pred_exp)
                holdout_score = root_mean_squared_log_error(test_data[TARGET_NAME].values, test_pred_exp)
                print(f"OOF RMSLE: {oof_score}", f"HOLDOUT RMSLE: {holdout_score}", sep='\n')

                results_df = test_data[['FlightDate', 'Dep_Airport', 'Tail_Number', 'Flight_Order', TARGET_NAME]].copy()
                results_df['Predicted_Arr_Delay'] = test_predictions.data[:, 0].round(1)
                results_df.to_csv('ModelAnswer.csv', index=False)

            print("Пайплайн успешно выполнен.")
        finally:
            sys.stdout = sys.__stdout__


        try:
            some_tail = results_df['Tail_Number'].iloc[0]
            df_plane = results_df[results_df['Tail_Number'] == some_tail].copy()
            if df_plane.shape[0] == 0:
                raise ValueError(f"Нет записей для самолёта {some_tail}")
            df_plane = df_plane.sort_values(by='FlightDate')
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Arr_Delay'], marker='o', label='Реальная задержка')
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Predicted_Arr_Delay'], marker='x', label='Прогноз')
            plt.title(f"Задержки для самолёта {some_tail}: факт vs прогноз")
            plt.xlabel("Дата рейса")
            plt.ylabel("Задержка (мин)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"delay_comparison_{some_tail}.png")
            some_tail = results_df['Tail_Number'].iloc[1]
            df_plane = results_df[results_df['Tail_Number'] == some_tail].copy()
            if df_plane.shape[1] == 0:
                raise ValueError(f"Нет записей для самолёта {some_tail}")
            df_plane = df_plane.sort_values(by='FlightDate')
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Arr_Delay'], marker='o',
                     label='Реальная задержка')
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Predicted_Arr_Delay'], marker='x',
                     label='Прогноз')
            plt.title(f"Задержки для самолёта {some_tail}: факт vs прогноз")
            plt.xlabel("Дата рейса")
            plt.ylabel("Задержка (мин)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"delay_comparison_{some_tail}.png")
            some_tail = results_df['Tail_Number'].iloc[2]
            df_plane = results_df[results_df['Tail_Number'] == some_tail].copy()
            if df_plane.shape[2] == 0:
                raise ValueError(f"Нет записей для самолёта {some_tail}")
            df_plane = df_plane.sort_values(by='FlightDate')
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Arr_Delay'], marker='o',
                     label='Реальная задержка')
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Predicted_Arr_Delay'], marker='x',
                     label='Прогноз')
            plt.title(f"Задержки для самолёта {some_tail}: факт vs прогноз")
            plt.xlabel("Дата рейса")
            plt.ylabel("Задержка (мин)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"delay_comparison_{some_tail}.png")
            some_tail = results_df['Tail_Number'].iloc[3]
            df_plane = results_df[results_df['Tail_Number'] == some_tail].copy()
            if df_plane.shape[3] == 0:
                raise ValueError(f"Нет записей для самолёта {some_tail}")
            df_plane = df_plane.sort_values(by='FlightDate')
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Arr_Delay'], marker='o',
                     label='Реальная задержка')
            plt.plot(pd.to_datetime(df_plane['FlightDate']), df_plane['Predicted_Arr_Delay'], marker='x',
                     label='Прогноз')
            plt.title(f"Задержки для самолёта {some_tail}: факт vs прогноз")
            plt.xlabel("Дата рейса")
            plt.ylabel("Задержка (мин)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"delay_comparison_{some_tail}.png")
        except Exception as e:
            print(f"Ошибка при построении графика реальной и предсказанной задержки для одного самолёта: {e}")

        try:
            plt.figure(figsize=(6, 6))
            plt.scatter(results_df['Arr_Delay'], results_df['Predicted_Arr_Delay'], alpha=0.5)
            max_val = max(results_df['Arr_Delay'].max(), results_df['Predicted_Arr_Delay'].max())
            plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray')  # линия y = x
            plt.title("Факт против прогноза (Arr_Delay vs Predicted_Arr_Delay)")
            plt.xlabel("Фактическая задержка (мин)")
            plt.ylabel("Прогноз задержки (мин)")
            plt.tight_layout()
            plt.savefig("scatter_actual_vs_predicted.png")
        except Exception as e:
            print(f"Ошибка при построении точечного графика 'факт против прогноза': {e}")

        except Exception as e:
            print(f"Ошибка при построении графика важности признаков: {str(e)}")
            # Дополнительная диагностика
            if 'feat_df' in locals():
                print(f"Тип объекта: {type(feat_df)}")
                if hasattr(feat_df, 'head'):
                    print(f"Пример данных:\n{feat_df.head()}")

if __name__ == "__main__":
    main()
