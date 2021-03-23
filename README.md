## No Name

This is a project using Deep Reinforcement Learning algorithm to train soccer robot in Robocup SSL with basic skills and tragedies.

- Algorithm

  Twin Delayed Deep Deterministic Policy Gradient.

  the paper is here [Addressing function approximation error in actor-critic methods](http://proceedings.mlr.press/v80/fujimoto18a.html)

- Simulation

  RoboCup small size league official simulator grSim

  the repository is here [grSim](https://github.com/RoboCup-SSL/grSim)

- Skills

  Go to ball skill

  rotating shoot skill

- Tactics

  run and shoot

#### Training instruction

- Go to ball skill

  train_findball.py

- Shoot skill

  train_shooting.py

- Switch reaction environment

  ssl-env is a completely kinematic environment for find ball skill

  my_env is a interface communicating with grSim

- Evaluation

  run_and_kick.py test it in grSim simulator

  real_car.py test skills in real world

- The environment use UDP communication protocol. The IP and interface would change according to your setting.



