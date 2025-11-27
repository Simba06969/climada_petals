"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define impact functions for heat waves.
"""

__all__ = ['ImpfHeatWave', 'MAX_WBGT_INTENSITY', 'DEFAULT_SIGMOID_STEEPNESS',
           'MDD_ZERO_THRESHOLD']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

# Maximum WBGT value in °C typically observed in extreme heat events
# Values above 45°C are extremely rare in real-world conditions
MAX_WBGT_INTENSITY = 45.0

# Default steepness parameter for sigmoid function
# Higher values create steeper transitions
DEFAULT_SIGMOID_STEEPNESS = 0.5

# Threshold below which mdd values are set to exactly zero
# to avoid numerical artifacts
MDD_ZERO_THRESHOLD = 0.01


class ImpfHeatWave(ImpactFunc):
    """Impact function for heat waves.

    This class provides impact functions based on the Wet Bulb Globe Temperature
    (WBGT), which is a composite temperature index accounting for temperature,
    humidity, and radiation exposure.

    The hazard type is 'HW' for heat wave.
    """

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'HW'
        self.intensity_unit = '°C'

    @classmethod
    def from_step_function(
        cls,
        impf_id=1,
        threshold=33.0,
        intensity_unit='°C',
        name='Step function heat wave'
    ):
        """Create a step function impact function for heat waves.

        The impact function uses a step function that activates at a given
        threshold. Below the threshold, the impact is 0. At and above the
        threshold, the impact is 100% (mdd=1).

        This is suitable for modeling binary outcomes such as heat-related
        mortality or work productivity loss.

        Parameters
        ----------
        impf_id : int, optional
            Impact function ID. Default: 1
        threshold : float, optional
            The WBGT threshold (in °C) at which impact begins.
            Default: 33.0°C, which corresponds to extreme heat stress.
        intensity_unit : str, optional
            Unit of the intensity. Default: '°C'
        name : str, optional
            Name of the impact function.

        Returns
        -------
        impf : ImpfHeatWave
            Impact function instance.

        Examples
        --------
        >>> impf = ImpfHeatWave.from_step_function(threshold=33.0)
        >>> impf.plot()

        Notes
        -----
        The step function is commonly used for heat wave impact assessment
        as described in literature. The threshold can be adjusted based on
        regional acclimatization levels.

        Common WBGT thresholds for heat stress:
        - 25°C: Caution zone begins
        - 28°C: Extreme caution
        - 32°C: Danger zone
        - 35°C: Extreme danger
        """
        impf = cls()
        impf.id = impf_id
        impf.name = name
        impf.intensity_unit = intensity_unit

        # Create intensity array with smooth transition at threshold
        epsilon = 0.1  # Small value for smooth step
        max_intensity = min(threshold + 20.0, MAX_WBGT_INTENSITY)
        impf.intensity = np.array([
            0.,
            threshold - epsilon,
            threshold,
            max_intensity
        ])
        impf.mdd = np.array([0., 0., 1., 1.])
        impf.paa = np.ones(len(impf.intensity))

        impf.check()
        return impf

    @classmethod
    def from_sigmoid_function(
        cls,
        impf_id=1,
        threshold=25.0,
        half_point=33.0,
        steepness=DEFAULT_SIGMOID_STEEPNESS,
        intensity_unit='°C',
        name='Sigmoid function heat wave'
    ):
        """Create a sigmoid-based impact function for heat waves.

        The impact function uses a sigmoid curve that smoothly transitions
        from 0 to 1 around the half_point value. This is more realistic
        than a step function for modeling gradual increases in heat-related
        impacts.

        The function is:
            mdd = 1 / (1 + exp(-steepness * (intensity - half_point)))

        where steepness controls the steepness of the transition.

        Parameters
        ----------
        impf_id : int, optional
            Impact function ID. Default: 1
        threshold : float, optional
            The WBGT value (in °C) below which impact is effectively 0.
            Default: 25.0°C
        half_point : float, optional
            The WBGT value (in °C) at which impact is 50%.
            Default: 33.0°C
        steepness : float, optional
            Controls the steepness of the sigmoid transition.
            Higher values create steeper transitions.
            Default: 0.5
        intensity_unit : str, optional
            Unit of the intensity. Default: '°C'
        name : str, optional
            Name of the impact function.

        Returns
        -------
        impf : ImpfHeatWave
            Impact function instance.

        Examples
        --------
        >>> impf = ImpfHeatWave.from_sigmoid_function(half_point=30.0)
        >>> impf.plot()
        """
        impf = cls()
        impf.id = impf_id
        impf.name = name
        impf.intensity_unit = intensity_unit

        # Create intensity array from threshold to maximum WBGT
        impf.intensity = np.linspace(threshold, MAX_WBGT_INTENSITY, 50)

        # Compute sigmoid function
        impf.mdd = 1 / (1 + np.exp(-steepness * (impf.intensity - half_point)))

        # Set values very close to 0 to exactly 0 to avoid numerical artifacts
        impf.mdd[impf.mdd < MDD_ZERO_THRESHOLD] = 0.0

        impf.paa = np.ones(len(impf.intensity))

        impf.check()
        return impf

    @classmethod
    def from_heat_mortality(
        cls,
        impf_id=1,
        threshold=28.0,
        intensity_unit='°C',
        name='Heat mortality function'
    ):
        """Create an impact function for heat-related mortality.

        This function is based on epidemiological studies of heat-related
        mortality. The relationship between temperature and mortality risk
        is typically J-shaped or U-shaped, with increased mortality at
        extreme temperatures.

        Parameters
        ----------
        impf_id : int, optional
            Impact function ID. Default: 1
        threshold : float, optional
            Temperature threshold (in °C) above which mortality risk increases.
            Default: 28.0°C
        intensity_unit : str, optional
            Unit of the intensity. Default: '°C'
        name : str, optional
            Name of the impact function.

        Returns
        -------
        impf : ImpfHeatWave
            Impact function instance.

        Notes
        -----
        This impact function is based on the relative risk approach commonly
        used in heat-health studies. The actual mortality rate depends on
        the baseline mortality and population vulnerability.
        """
        impf = cls()
        impf.id = impf_id
        impf.name = name
        impf.intensity_unit = intensity_unit

        # Create intensity array
        impf.intensity = np.array([0., threshold, threshold + 5,
                                   threshold + 10, threshold + 15, 50.])

        # Relative damage values based on heat mortality studies
        # Values increase exponentially above threshold
        impf.mdd = np.array([0., 0., 0.2, 0.5, 0.8, 1.0])

        impf.paa = np.ones(len(impf.intensity))

        impf.check()
        return impf
