import math
from functools import lru_cache

import numpy as np
from common.handlers.base import MessageHandler
from common.models.oil_data import OilData
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from vlp_service.schemas import PVT, Inclinometry, Pipeline, Tubing, VlpRequest


class CalculateService(MessageHandler):
    def __init__(self):
        super().__init__()

    def calculate_vlp(self, request: VlpRequest) -> OilData:
        """Рассчитать координаты кривой VLP."""
        debits_volume = np.linspace(0.001, 400, 7)
        pressure = np.empty_like(debits_volume)

        for count, debit_volume in enumerate(debits_volume):

            pressure[count] = self.calculate_downhole_pressure(
                request.inclinometry,
                request.casing,
                request.tubing,
                request.pvt,
                request.p_wh,
                request.geo_grad,
                request.h_res,
                debit_volume,
            )

        return OilData.model_validate({"q_liq": debits_volume.tolist(), "p_wf": pressure.tolist()})

    def calculate_downhole_pressure(
        self,
        inclinometry: Inclinometry,
        casing: Pipeline,
        tubing: Tubing,
        pvt: PVT,
        buffer_pressure: float,
        geothermal_gradient: float,
        perforation_depth: float,
        debit_volume: float,
    ) -> float:
        """
        Расчитать забойное давление в скважине.
        Parameters
        ----------
        :param inclinometry: данные инклинометрии
        :param casing: данные по ЭК
        :param tubing: данные по НКТ
        :param pvt: данные по свойствам флюидов
        :param buffer_pressure: буферное давление, атм
        :param geometric_gradient: геотермический градиент, C/100 м
        :param perforation_depth: глубина верхних дыр перфорации
        :param debit_volume: дебит скважины
        :return: забойное давление, атм
        """
        pvt.t_res = self._convert_сelsius_to_kelvin(pvt.t_res)
        buffer_pressure = self._convert_pressure(buffer_pressure, "atm", "pa")
        pvt.wct = pvt.wct / 100

        # Интерполяционный полином инклинометрии
        incl = self.interpolate_func(tuple(inclinometry.MD), tuple(inclinometry.TVD))

        # Расчёт давления на приеме (конце НКТ)
        admission_pressure = self.calculate_pressure(
            tubing.d,
            0,
            tubing.h_mes,
            incl,
            geothermal_gradient,
            pvt,
            buffer_pressure,
            incl(perforation_depth).item(),
            debit_volume,
        )

        # Расчёт забойного давления
        p_wf = self.calculate_pressure(
            casing.d,
            tubing.h_mes,
            perforation_depth,
            incl,
            geothermal_gradient,
            pvt,
            admission_pressure,
            incl(perforation_depth).item(),
            debit_volume,
        )
        return self._convert_pressure(p_wf, "pa", "atm")

    @staticmethod
    def _convert_сelsius_to_kelvin(temperature: int | float) -> int | float:
        """Конвертировать градусы Цельсия в Кельвины."""
        return temperature + 273.15

    @staticmethod
    def _convert_pressure(
        pressure: float, current_measurement_unit: str, expected_measurement_unit: str
    ) -> int | float:
        """Конвертировать давление из паскалей в атмосферы, либо наоборот."""
        if current_measurement_unit == "atm":
            pressure *= 101325

        if expected_measurement_unit == "atm":
            pressure /= 101325

        return pressure

    @staticmethod
    @lru_cache(maxsize=1024)
    def interpolate_func(inclinometry_md: tuple, inclinometry_tvd: tuple) -> interp1d:
        """
        Интерполировать функцию инклинометрии.
        """
        return interp1d(inclinometry_md, inclinometry_tvd, fill_value="extrapolate")

    def calculate_pressure(
        self,
        diameter: int | float,
        initial_depth: int | float,
        boundary_depth: int | float,
        inclinometry,
        geothermal_gradient: int | float,
        pvt: PVT,
        input_pressure: int | float,
        tvd_res: interp1d,
        debit_volume: float,
    ):
        """
        Вычислить давление на участке трубы.
        Parameters
        ----------
        :param diameter: диаметр трубы, м
        :param initial_depth: начальная глубина, м
        :param boundary_depth: граничная глубина, м
        :param inclinometry: интерполяционный полином инклинометрии
        :param geothermal_gradient: геотермический градиент
        :param pvt: данные по свойствам флюидов
        :param input_pressure: входное давление в трубу, атм
        :param tvd_res: интерполяционный полином инклинометрии, безразмерный
        :param debit_volume: дебиты жидкости, м3/сут
        :return: выходное давление из трубы, атм
        """
        tvd_cur = inclinometry(initial_depth).item()

        result = solve_ivp(
            self._calculate_gradient,
            t_span=(initial_depth, boundary_depth),
            y0=[
                input_pressure,
                self.calculate_temperature(geothermal_gradient, pvt.t_res, tvd_res, tvd_cur),
            ],
            method="RK23",
            args=(diameter, geothermal_gradient, pvt, inclinometry, debit_volume),
        )

        return result.y[0][-1]

    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_temperature(
        geothermal_gradient: int | float, t_res: float, tvd_res: float, tvd_cur: float
    ) -> float:
        """
        Рассчитать температуру.
        Parameters
        ----------
        :param geothermal_gradient: геотермический градиент, C/100 м
        :param t_res: пластовая температура, С
        :param tvd_res: интерполяционный полином инклинометрии, безразмерный
        :param tvd_cur: интерполяционный полином инклинометрии от глубиины, безразмерный
        """
        return t_res - geothermal_gradient * (tvd_res - tvd_cur) / 100

    @staticmethod
    def _calculate_mixture_properties(
        pressure: float, temperature: float, pvt: PVT, debit_volume: float
    ) -> tuple[float, float, float]:
        """Вернуть тестовые данные для свойств смеси."""
        return 1, 0.82, 40

    def _calculate_gradient(
        self,
        depth: float,
        pressure_temperature: list[float, float],
        pipe_diameter: float,
        geothermal_gradient,
        pvt: PVT,
        incl: interp1d,
        debit_volume: float,
    ) -> tuple[float, float]:
        """Вычислить итоговый градиент."""
        angle = self._calculate_angle(incl, depth, depth - 0.0001)

        mixture_volume, mixture_density, mixture_viscosity = self._calculate_mixture_properties(
            pressure_temperature[0], pressure_temperature[1], pvt, debit_volume
        )

        mixture_speed = self._calculate_mixture_speed(mixture_volume, pipe_diameter)
        reynolds_number = self._calculate_reynolds_number(
            pipe_diameter, mixture_density, mixture_speed, mixture_viscosity
        )

        friction_coefficient = self._calculate_friction_coefficient(reynolds_number, 0.0001)

        friction_gradient = self._calculate_friction_gradient(
            friction_coefficient, mixture_density, mixture_speed, pipe_diameter
        )

        gravity_gradient = self._calculate_gravity_gradient(mixture_density, angle)

        return friction_gradient + gravity_gradient, geothermal_gradient / 100

    def _calculate_angle(
        self, incl, first_measured_depth: float, second_measured_depth: float
    ) -> float:
        """
        Расчитать угол по интерполяционной функции траектории по измеренной глубине.
        Parameters
        ----------
        :param first_measured_depth: measured depth 1, м
        :param second_measured_depth: measured depth 2, м
        :return: угол к горизонтали, градусы
        """
        return (
            np.degrees(
                np.arcsin(
                    self._calc_sin_angle(incl, first_measured_depth, first_measured_depth + 0.001)
                )
            )
            if second_measured_depth == first_measured_depth
            else np.degrees(
                np.arcsin(self._calc_sin_angle(incl, first_measured_depth, second_measured_depth))
            )
        )

    @staticmethod
    def _calc_sin_angle(incl, first_measured_depth: float, second_measured_depth: float) -> float:
        """
        Расчитать синус угла с горизонталью по интерполяционной функции скважины.
        Parameters
        ----------
        :param first_measured_depth: measured depth 1, м
        :param second_measured_depth: measured depth 2, м
        :return: синус угла к горизонтали
        """
        return (
            0
            if second_measured_depth == first_measured_depth
            else min(
                (incl(second_measured_depth).item() - incl(first_measured_depth).item())
                / (second_measured_depth - first_measured_depth),
                1,
            )
        )

    @staticmethod
    def _calculate_gravity_gradient(mixture_density: float, angle: float) -> float:
        """Вычислить гравитационный градиент."""
        return mixture_density * 9.81 * math.sin(angle / 180 * math.pi)

    @staticmethod
    def _calculate_friction_gradient(
        friction_coefficient: float,
        mixture_density: float,
        mixture_speed: float,
        pipe_diameter: float,
    ) -> float:
        """Вычислить градиент трения."""
        return friction_coefficient * mixture_density * mixture_speed**2 / (2 * pipe_diameter)

    @staticmethod
    def _calculate_mixture_speed(volume_mixture: float, pipe_diameter: float) -> float:
        """Рассчитать скорость смеси."""
        return volume_mixture / (math.pi * pipe_diameter**2 / 4)

    @staticmethod
    def _calculate_friction_coefficient(reynolds_number: float, effective_pipe_roughness: float):
        """
        Вычислисть коэффициент трения на основе числа Рейнольдса и шероховатости трубы.

        Parameters
        ----------
        :param reynolds_number: число Рейнольдса, безразмерн.
        :param effective_pipe_roughness: эффективная шероховатость трубы, мкм
        :return: коэффициент трения, безразмерн.
        """
        if reynolds_number == 0:
            friction_coefficient = 0
        # рассчитать коэффициент трения при ламинарном потоке (в методичке указано, что ламинарным
        # является поток, с числом Рейнольдса <= 2000, однако множество других источников указывают,
        # что ламинарным является поток с числом Рейнольдса менее, либо равным 2300,
        # в связи с этим внес правки)
        elif reynolds_number <= 2300:
            friction_coefficient = 64 / reynolds_number
        else:
            # рассчитать начальный коэффициент трения
            initial_friction_coefficient = (
                2
                * math.log10(
                    0.5405405405405405 * effective_pipe_roughness
                    - 5.02
                    / reynolds_number
                    * math.log10(
                        0.5405405405405405 * effective_pipe_roughness + 13 / reynolds_number
                    )
                )
            ) ** -2

            if reynolds_number <= 4000:
                # рассчитать коэффициент трения при переходном потоке
                # исправил min_re, т.к. выше взяли за значение, являющееся пограничным, 2300
                min_re = 2300
                max_re = 4000
                f_lam = 0.032
                friction_coefficient = f_lam + (reynolds_number - min_re) * (
                    initial_friction_coefficient - f_lam
                ) / (max_re - min_re)
            else:
                friction_coefficient = initial_friction_coefficient
                count = 0
                while True:
                    # в соответствии с методичкой исправлена степень, в которую возводится
                    # коэффициент трения
                    new_friction_coefficient = (
                        1.74
                        - 2
                        * math.log10(
                            2 * effective_pipe_roughness
                            + 18.7 / (reynolds_number * friction_coefficient**2)
                        )
                    ) ** -2
                    count = count + 1
                    error = (
                        abs(new_friction_coefficient - friction_coefficient)
                        / new_friction_coefficient
                    )
                    friction_coefficient = new_friction_coefficient
                    # прекратить цикл, если ошибка достаточно незначительта, либо превышено
                    # максимальное количество попыток
                    if error <= 0.0001 or count > 19:
                        break

        return friction_coefficient

    @staticmethod
    def _calculate_reynolds_number(
        pipe_diameter: float, mixture_density: float, mixture_speed: float, mixture_viscosity: float
    ) -> float:
        """
        Вычислисть число Рейнольдса.
        Parameters
        ----------
        :param pipe_diameter: диаметр трубы, м
        :param mixture_density: плотность смеси, кг/м3
        :param mixture_speed: скорость смеси, м/с
        :param mixture_viscosity: вязкость смеси, сПз
        :return: число Рейнольдса, безразмерн.
        """
        return (
            1000
            * mixture_density
            * mixture_speed
            * pipe_diameter
            / max(mixture_viscosity, 0.000001)
        )
