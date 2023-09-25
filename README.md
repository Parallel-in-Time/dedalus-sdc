# IMEX Spectral Deferred Correction (SDC) solver for Dedalus

:scroll: _This project focus on the implementation of an IMEX-SDC solver for the [Dedalus framework](https://dedalus-project.readthedocs.io/en/latest/index.html) that can be used to solve partial differential equations using spectral methods (Fourier based, not polynomial)._

Spectral Deferred Correction were originally developed by [Dutt and Greengard](#dutt2000spectral) in order to provide generic time integration method with arbitrary high order of accuracy.
They received a particular attention with the development of time-parallel methods, in particular with 
[Parallel Deferred Correction](#guibert2007parallel) or 
[PFASST](#emmett2012toward) allowing time-parallelization across the time-steps, or [Parallel Diagonal SDC](#speck2018parallelizing) allowing parallelization across the stages within a time-step.
This code intend to investigate the **accuracy** and **performance** of those methods when used in combination with the spectral method of Dedalus.

The IMEX-SDC solver can be used to solve initial value problem of the form :

$$
\mathcal{M} \cdot \partial_t \mathcal{X} + \mathcal{L} \cdot \mathcal{X} = \mathcal{F}(\mathcal{X}, t)
$$

where $\mathcal{L}$ is a linear term, solved implicitly with Dedalus internal solvers, and $\mathcal{F}(\mathcal{X}, t)$ a (non-)linear term solved explicitly.
See [Dedalus documentation](https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html#Problem-formulations) for more details and how to set up all those terms.

## Structure of the repository

- [dedalus_sdc](./dedalus_sdc/) : main code folder
- [scripts](./scripts/) : example scripts using IMEX-SDC
- [dedalus_install.md](./dedalus_install.md) : instruction to install Dedalus

## Acknowledgements

This project has received funding from the [European High-Performance Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU)
under grant agreement No 955701 ([Time-X](https://www.timex-eurohpc.eu/)).
The JU receives support from the European Union’s Horizon 2020 research and innovation programme and Belgium, France, Germany, and Switzerland.
This project also received funding from the
[German Federal Ministry of Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html) (BMBF) grant 16HPC048.

<p align="center">
  <img src="./doc/images/EuroHPC.jpg" width="35%"/> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doc/images/LogoTime-X.png" width="25%"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doc/images/BMBF_gefoerdert_2017_en.jpg" width="20%" />
</p>

## Bibliography

<a id="dutt2000spectral">[Dutt & Greengard, 2000]</a> _Dutt, A., Greengard, L., & Rokhlin, V. (2000). [Spectral deferred correction methods for ordinary differential equations](https://link.springer.com/content/pdf/10.1023/A:1022338906936.pdf). BIT Numerical Mathematics, 40, 241-266._

<a id="guibert2007parallel">[Guibert & Tromeur-Dervout, 2007]</a>_Guibert, D., & Tromeur-Dervout, D. (2007). [Parallel deferred correction method for CFD problems.](./biblio/guibert2007parallel.pdf) In Parallel Computational Fluid Dynamics 2006 (pp. 131-138). Elsevier Science BV._

<a id="emmett2012toward">[Emmett & Minion, 2012]</a> _Emmett, M., & Minion, M. (2012). [Toward an efficient parallel in time method for partial differential equations](https://projecteuclid.org/journals/communications-in-applied-mathematics-and-computational-science/volume-7/issue-1/Toward-an-efficient-parallel-in-time-method-for-partial-differential/10.2140/camcos.2012.7.105.pdf). Communications in Applied Mathematics and Computational Science, 7(1), 105-132._

<a id="speck2018parallelizing">[Speck, 2018]</a>
_Speck, R. (2018). [Parallelizing spectral deferred corrections across the method.](https://link.springer.com/article/10.1007/s00791-018-0298-x) Computing and visualization in science, 19, 75-83._