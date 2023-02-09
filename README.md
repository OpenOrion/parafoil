<p align="center">
    <img src="./assets/logo.svg" alt="drawing" width="500"/>
</p>

<p align="center">the open source parametric airfoil generator</p>

<p align="center">
    <a href="https://discord.gg/H7qRauGkQ6">
        <img src="https://img.shields.io/discord/913193916885524552?logo=discord"
            alt="chat on Discord">
    </a>
    <a href="https://www.patreon.com/turbodesigner">
        <img src="https://img.shields.io/badge/dynamic/json?color=%23e85b46&label=Patreon&query=data.attributes.patron_count&suffix=%20patrons&url=https%3A%2F%2Fwww.patreon.com%2Fapi%2Fcampaigns%2F9860430"
            alt="donate to Patreon">
    </a>
</p>



# About
ParaFoil is a declarative tool that parametrically generates turbomachinery and aircraft airfoils and passages with B-splines. The purpose is to generate airfoil shapes from different parameters during fluid simulation optimizations.


# Install
```
pip install git+https://github.com/Turbodesigner/parafoil.git#egg=parafoil
```

# Example

## Airfoil
```python
import numpy as np
from parafoil import Airfoil, PassageBuilder

airfoil = Airfoil(
    inlet_angle=np.radians(40.0),
    outlet_angle=np.radians(20.0),
    stagger_angle=np.radians(0.0),
    upper_thick_dist=[0.05, 0.2, 0.2, 0.05, 0.01],
    lower_thick_dist=[0.05, 0.2, 0.2, 0.05, 0.01],
    leading_prop=0.5,
    trailing_prop=0.5,
    chord_length=1.44
)
python(airfoil.coords)
airfoil.visualize(include_camber_ctrl_pnts=True)
```
![Airfoil](./assets/airfoil.png)

## Passage
```python
passage = PassageBuilder.airfoil_row_passage(
    airfoil=airfoil,
    spacing=1.0 * airfoil.chord_length,
    leading_edge_gap=0.25,
    trailing_edge_gap=0.25,
    num_blades=1
)

passage.visualize()
```

![Passage](./assets/passage.png)


# Devlopement Setup
```
git clone https://github.com/Turbodesigner/parafoil.git
cd parafoil
pip install -r requirements_dev.txt
```

# Help Wanted
Right now there are some items such as CFD meshing and adding more shaping. Please join the [Discord](https://discord.gg/H7qRauGkQ6) for project communications and collaboration. Please consider donating to the [Patreon](https://www.patreon.com/turbodesigner) to support future work on this project.

