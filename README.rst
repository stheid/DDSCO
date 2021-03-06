======================================================
Distributed Delayed Stochastic Constraint Optimization
======================================================

|license|

.. |license| image:: https://img.shields.io/github/license/stheid/DDSCO
    :target: LICENSE


This code provides experimental results and a simple algorithm to solve distributed constrained optimization tasks.
The experiment implemented here will simulate robot a robot swarm to arrange on a line or a circle.
The objective functions have been implemented in torch_ such that automatic gradient calculation could be used.

.. _`torch`: https://pytorch.org/

|line| |circ|

.. |line| image:: results/line.gif
    :target: results/line.gif
.. |circ| image:: results/circle.gif
    :target: results/circle.gif


Overview
--------
- :code:`scripts` contain usage files
- :code:`ddsco` contains the implementation
- :code:`results` contains experimental results

Installation
------------
**Disclaimer**: The code probably doesn't run without modifications on Windows.
It should work on any standard Linux distribution.

* Install Python package::

  $ pip install .

Results
-------
|circ-changing_comm|

The convergence speed is strongly effected by the quality of the communication channel.
In the left video, the communication channel has higher transmission probabilities and hence the overall convergence is speed up.

.. |circ-changing_comm| image:: results/circle-channel_quali.gif
    :target: results/circle-channel_quali.mp4