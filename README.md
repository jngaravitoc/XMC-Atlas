# XMC Atlas

Library of available MW-LMC potential models

### Available models:

| Model | Reference |  Expansion class | Coefficients format | 
--------| ----------|------------------| --------------------|
| GC21 |  [Garavito-Camargo et. al., 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...919..109G/abstract)  | Hernquist | Gala & EXP |
| L21  |  [Lilleengen et. al., 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518..774L/abstract)   | Empirical | EXP |
| V23  |  [Vasiliev et. al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2279V/abstract)  | Spline and Hernquist | AGAMA |  




### How to contribute:
----------------------- 
To contribute, please make a pull request (PR) including the following information:

- Fill out the model parameters with the following information:
  [] Orbit files of the LMC and the MW
  [] Halo properties of the LMC and the MW represented as a Gala potential representation
  [] Reference of the paper where the models are presented
- Add the new model in the model folder with the following subdirectories:
  - coefficients/
  - model_params/
  - orbits/
- Add code with an example of how to read the coefficients and plot density fields in the examples folder


### Authors:
-------------

- Nico Garavito-Camargo
- Sophia Lilleengen
- Mike Petersen
