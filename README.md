# The "Friendly" Patrick Bateman model

<img width="549" height="196" alt="image" src=../addenda/patrick Bateman.jpgalt="1-u-Ms06-ROBVq0-Q-5xcrc-A" border="0"/>

## Introduction

        This project simulates a long‑term portfolio management system for US stocks using
    Deep Deterministic Policy Gradient (DDPG). The main focus is a **pure price‑based DDPG agent**, 
    which learns a trading policy from historical prices and technical features.

        On top of that, there is an **experimental extension** that injects a small language model (IBM Granite) 
    as a weekly news‑sentiment feature. The goal is to explore whether coarse, low‑frequency sentiment
    signals can improve the behaviour of a long‑horizon RL agent, but this part should be treated as an experimental add‑on, 
    not the default recommended setup.

