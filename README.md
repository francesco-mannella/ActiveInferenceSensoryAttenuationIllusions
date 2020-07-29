ActiveInferenceSensoryAttenuationIllusions


Implementation of the model described in :

## Active inference, sensory attenuation and illusions.
### Brown, H., Adams, R. A., Parees, I., Edwards, M., & Friston, K. (2013).
### Cognitive Processing, 14(4), 411–427.
### https: // doi.org/10.1007/s10339-013-0571-3

 ## Generative Process

 $$
\begin{array}{llll}
    \mathbf{s} &=& \left[\begin{array}{c} s_p \\ s_s\end{array}\right] &=&
    \left[\begin{array}{c} x_i \\ x_i + v_e \end{array}\right] +
    \boldsymbol{\omega}_s &&&&\\
 \dot{x} &=& \dot{x}_i  &=& \sigma{\alpha}
    \frac{1}{4} x_i + \omega_x\\
 \boldsymbol{\omega}_s &\sim& \mathcal{N}(\mathbf{0},
      e^{-8} \mathbf{I})\\
 \omega_x &\sim& \mathcal{N}(0, e^{-8})
 \end{array}
 $$

 ## Generative Model
$$
\begin{array}{lllll}
\mathbf{s} &= & \left[\begin{array}{c} s_p \\ s_s\end{array}\right] &= &
\left[\begin{array}{c} x_i \\ x_i + x_e \end{array}\right] +
\boldsymbol{\omega}_s & & & &\\
    \mathbf{\dot{x}} &= & \left[\begin{array}{c} \dot{x_i} \\ \dot{x}_e
    \end{array}\right] & = &
\left[\begin{array}{c} \nu_i -\frac{1}{4} \dot{x}_i\\
      \nu_e -\frac{1}{4} \dot{x}_e\end{array}\right] +
\boldsymbol{\omega}_s & & & &\\
\boldsymbol{\nu} &&&=& \left[\begin{array}{c} \nu_i \\ \nu_e
  \end{array}\right] + \boldsymbol{\omega}_{\nu}\\
\boldsymbol{\omega}_s &\sim&
   \mathcal{N}(\mathbf{0}, e^{\pi} \mathbf{I})\\
\boldsymbol{\omega}_x &\sim&
   \mathcal{N}(\mathbf{0}, e^{-4} \mathbf{I})\\
\boldsymbol{\omega}_\nu &\sim&
   \mathcal{N}(\mathbf{0}, e^{-6} \mathbf{I})\\
\pi &=& 8-\gamma\sigma(x_i + \nu_i)
\end{array}
$$


 ## Variational Laplace Encoded Free Energy
 $$
 \begin{array}{lll}
  F &=& -log(P(s, \boldsymbol{\mu}_x, \boldsymbol{\mu}_\boldsymbol{\nu})) + C \\
  &=& -log(P(s|\boldsymbol{\mu}_x)P(\dot{\boldsymbol{\mu}}_{x}|\boldsymbol{\mu}_x,
    \boldsymbol{\mu}_\boldsymbol{\nu})P(\dot{\boldsymbol{\mu}}_
      \boldsymbol{\nu}|\boldsymbol{\mu}_\boldsymbol{\nu})) + C  \\
  &=& -log(
    \mathcal{N}( g(\boldsymbol{\mu}_x),  e^{-\pi}\mathbf{I})
    \mathcal{N}( f(\boldsymbol{\mu}_x,\boldsymbol{\mu}_\nu), e^{-4}\mathbf{I})
    \mathcal{N}(\boldsymbol{\mu}_{\nu}, e^{-6}\mathbf{I})
  ) + C  \\

 \end{array}
 $$
