#!/usr/bin/env python

"""
Module for calculations related to the contrast transfer function (CTF).
"""

from typing import Tuple, Union, TypeVar, Dict

import numpy as np
import torch

from .spectral import get_freq

from functools import lru_cache

Tensor = TypeVar('torch.tensor')


class ContrastTransferFunction:
    def __init__(
            self,
            voltage: float,
            spherical_aberration: float = 0.,
            amplitude_contrast: float = 0.,
            phase_shift: float = 0.,
            b_factor: float = 0.,
            pixel_size: float = None
    ) -> None:
        """
        Initialization of the CTF parameter for an optics group.
        :param voltage: Voltage
        :param spherical_aberration: Spherical aberration
        :param amplitude_contrast: Amplitude contrast
        :param phase_shift: Phase shift
        :param b_factor: B-factor
        :param pixel_size: Pixel size
        """

        if voltage <= 0:
            raise RuntimeError(
                f"Invalid value ({voltage}) for voltage of optics group {id}."
            )

        self.voltage = voltage
        self.spherical_aberration = spherical_aberration
        self.amplitude_contrast = amplitude_contrast
        self.phase_shift = phase_shift
        self.b_factor = b_factor
        self.pixel_size = pixel_size
        # Adjust units
        spherical_aberration = spherical_aberration * 1e7
        voltage = voltage * 1e3

        # Relativistic wave length
        # See http://en.wikipedia.org/wiki/Electron_diffraction
        # lambda = h/sqrt(2*m*e) * 1/sqrt(V*(1+V*e/(2*m*c^2)))
        # h/sqrt(2*m*e) = 12.2642598 * 10^-10 meters -> 12.2642598 Angstrom
        # e/(2*m*c^2)   = 9.78475598 * 10^-7 coulombs/joules
        lam = 12.2642598 / np.sqrt(voltage * (1. + voltage * 9.78475598e-7))

        # Some constants
        self.c1 = -np.pi * lam
        self.c2 = np.pi / 2. * spherical_aberration * lam ** 3
        self.c3 = -b_factor/4.
        self.c4 = \
            np.arctan(
                amplitude_contrast / np.sqrt(1-amplitude_contrast**2)
            )

    def __call__(
            self,
            shape: Tuple[int, ...],
            u: Tensor,
            v: Tensor = None,
            angle: Tensor = None,
            pixel_size: float = None,
            b_factor: Tensor = None,
            rfft: bool = True,
            center: bool = True,
            phase_shift: Tensor = None,
    ) -> Tensor:
        """
        Get the CTF in an numpy array, the size of freq_x or freq_y.
        Generates a Numpy array or a Torch tensor depending on the object type
        on freq_x and freq_y passed to the constructor.
        :param u: the U defocus
        :param v: the V defocus
        :param angle: the azimuthal angle defocus (degrees)
        :param grid_size: the side of the box
        :param pixel_size: pixel size
        :param b_factor: per CTF B-factor, overwritting the global one included in constructor
        :param rfft: Only consider the hermitian half
        :return: Numpy array or Torch tensor containing the CTF
        :param phase_shift: phase shift in degrees, overwriting the global one included in constructor
        """

        if len(shape) == 2 and v is None:
            v = u
            angle = torch.zeros_like(u)

        if pixel_size is None:
            assert self.pixel_size is not None, "Pixel size is not set"
            pixel_size = self.pixel_size
        
        u = u.view(-1)
        freq = self._get_freq(
            shape=shape, 
            pixel_size=pixel_size, 
            rfft=rfft, 
            center=center,
            device=u.device
        )

        # 1 Dimensions
        if len(shape) == 1:
            n2, n4 = freq
            gamma = self.c1 * u[:, None] * n2[None] + self.c2 * n4[None] - self.c4

            if phase_shift is not None:
                phase_shift = phase_shift.view(-1)
                gamma -= phase_shift[:, None] * np.pi / 180.

            ctf = -torch.sin(gamma)
            if b_factor is not None:
                c3 = -b_factor/4.
                ctf *= torch.exp(c3 * n4[None])
            elif self.c3 != 0:
                ctf *= torch.exp(self.c3 * n4[None])


        # 2 Dimensions
        elif len(shape) == 2:
            v = v.view(-1)
            xx, yy, xy, n4 = freq

            angle = angle * np.pi / 180
            acos = torch.cos(angle)
            asin = torch.sin(angle)
            acos2 = torch.square(acos)
            asin2 = torch.square(asin)

            """
            Outline of math for following three lines of code
            Q = [[sin cos] [-sin cos]] sin/cos of the angle
            D = [[u 0] [0 v]]
            A = Q^T.D.Q = [[Axx Axy] [Ayx Ayy]]
            Axx = cos^2 * u + sin^2 * v
            Ayy = sin^2 * u + cos^2 * v
            Axy = Ayx = cos * sin * (u - v)
            defocus = A.k.k^2 = Axx*x^2 + 2*Axy*x*y + Ayy*y^2
            """

            xx_ = (acos2 * u + asin2 * v)[:, None, None] * xx[None]
            yy_ = (asin2 * u + acos2 * v)[:, None, None] * yy[None]
            xy_ = (acos * asin * (u - v))[:, None, None] * xy[None]

            gamma = self.c1 * (xx_ + 2. * xy_ + yy_) + self.c2 * n4[None] - self.c4

            if phase_shift is not None:
                phase_shift = phase_shift.view(-1)
                gamma -= phase_shift[:, None, None] * np.pi / 180.

            ctf = -torch.sin(gamma)
            if b_factor is not None:
                c3 = -b_factor/4.
                ctf *= torch.exp(c3 * n4[None])
            elif self.c3 != 0:
                ctf *= torch.exp(self.c3 * n4[None])
        
        return ctf

    @staticmethod
    @lru_cache(maxsize=5, typed=False)
    def _get_freq(
            shape: Tuple[int, ...],
            pixel_size: float,
            rfft: bool = False,
            center: bool = True,
            device="cpu"
    ) -> Union[
            Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]],
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
         ]:
        freq = get_freq(
            shape=shape,
            pixel_size=pixel_size,
            rfft=rfft,
            device=device,
            center=center
        )
        if len(shape) == 1:
            n2 = freq**2
            n4 = n2**2
            return n2, n4
        elif len(shape) == 2:
            freq_x, freq_y = freq
            xx = freq_x**2
            yy = freq_y**2
            xy = freq_x * freq_y
            n4 = (xx + yy)**2  # Norms squared^2

            return xx, yy, xy, n4


    def get_state_dict(self) -> Dict:
        return {
            "type": "ContrastTransferFunction",
            "version": "0.0.1",
            "voltage": self.voltage,
            "spherical_aberration": self.spherical_aberration,
            "amplitude_contrast": self.amplitude_contrast,
            "phase_shift": self.phase_shift,
            "b_factor": self.b_factor
        }

    @staticmethod
    def load_from_state_dict(state_dict):
        if "type" not in state_dict or state_dict["type"] != "ContrastTransferFunction":
            raise TypeError("Input is not an 'ContrastTransferFunction' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ContrastTransferFunction instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            return ContrastTransferFunction(
                voltage=state_dict['voltage'],
                spherical_aberration=state_dict['spherical_aberration'],
                amplitude_contrast=state_dict['amplitude_contrast'],
                phase_shift=state_dict['phase_shift'],
                b_factor=state_dict['b_factor'],
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
