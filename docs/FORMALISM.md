# Minimal formal scaffold

- `z_t = encoder(x_t, h_prev)`
- `s_t = self_model(z_t, s_prev, h_prev)`
- `L_t = loss_fn(z_t, s_t, goals_t)`
- `v_t = -grad_z(L_t)`
- `Phi_t = integration_metric(z_t, z_prev)`
- `A_t = access_metric(model_state)`
- `q_t = interface_synthesis(z_t, s_t, v_t, Phi_t, A_t)`
- `chi_t = sigma(a*Phi_t + b*MI(s_t,z_t) + c*||v_t|| - d*A_t)`
- `u_t = policy(z_t, s_t, q_t)`
