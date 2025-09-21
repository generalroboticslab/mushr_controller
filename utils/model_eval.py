import matplotlib.pyplot as plt
import numpy as np
import os

def multi_step_predict(model, init_obs, actions, type):
    obs = init_obs
    preds = []

    for t in range(actions.shape[0]):
        act = actions[t:t+1]
        if type == "sym":
            next_obs = model.forward(np.concatenate((obs, act), axis=-1))
        elif type == "mlp":
            next_obs = model.predict(obs, act).reshape(1, -1)
        preds.append(next_obs)
        obs = next_obs  # feed it back in

    return np.stack(preds, axis=1)[0]    

def evaluate_model(
    dynamics_model,
    replay_buffer,
    indices=np.arange(1, 400),
    save_dir=".",
    type="sym",
):
    """
    Evaluate current and previous dynamics models on one-step and autoregressive predictions.
    """
    save_dir = os.path.join(save_dir, "model_eval_figs")
    os.makedirs(save_dir, exist_ok=True)
    curr_model = dynamics_model.model
    test_batch = replay_buffer._batch_from_indices(indices)

    # One-step prediction
    x_input = np.concatenate((test_batch.obs, test_batch.action), axis=-1)
    
    # Handle different model structures for MuSHR vs Crazyflie
    try:
        # Try to use model_A if it exists (Crazyflie style)
        if hasattr(curr_model, 'model_A'):
            prev_pred_next = curr_model.model_A.forward(x_input)
        else:
            # For MuSHR models without model_A, use the base model
            prev_pred_next = curr_model.forward(x_input)
    except Exception as e:
        print(f"Warning: Could not get previous predictions: {e}")
        prev_pred_next = np.zeros_like(test_batch.next_obs)

    if type == "sym":
        curr_pred_next = curr_model.forward(x_input)
    elif type == "mlp":
        try:
            # Handle different MLP prediction methods
            if hasattr(curr_model, 'predict'):
                curr_pred_next = curr_model.predict(test_batch.obs, test_batch.action)
            else:
                # Fallback to forward method
                curr_pred_next = curr_model.forward(x_input)
            
            # Handle different output shapes
            if hasattr(curr_pred_next, 'mean'):
                curr_pred_next = curr_pred_next.mean(axis=0)
            elif len(curr_pred_next.shape) > 2:
                curr_pred_next = curr_pred_next.reshape(curr_pred_next.shape[0], -1)
        except Exception as e:
            print(f"Warning: MLP prediction failed: {e}")
            curr_pred_next = np.zeros_like(test_batch.next_obs)

    one_step_err = np.mean(np.abs(test_batch.next_obs - curr_pred_next), axis=0)
    prev_one_step_err = np.mean(np.abs(test_batch.next_obs - prev_pred_next), axis=0)

    print("One-step error:", one_step_err)
    print("Previous one-step error:", prev_one_step_err)

    # Multi-step (autoregressive) prediction
    init_obs = test_batch.obs[0:1]
    actions = test_batch.action
    gt_next_obs = test_batch.next_obs

    try:
        # Handle different model structures
        if hasattr(curr_model, 'model_A'):
            prev_auto_preds = multi_step_predict(curr_model.model_A, init_obs, actions, "sym")
        else:
            prev_auto_preds = multi_step_predict(curr_model, init_obs, actions, "sym")
        
        curr_auto_preds = multi_step_predict(dynamics_model, init_obs, actions, type)
        
        prev_auto_err = np.sum(np.abs(gt_next_obs - prev_auto_preds), axis=1)
        curr_auto_err = np.sum(np.abs(gt_next_obs - curr_auto_preds), axis=1)

        print("Auto-regressive error (current):", curr_auto_err)
        print("Auto-regressive error (previous):", prev_auto_err)

        # Plot per-dimension autoregressive error
        for i in range(gt_next_obs.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.plot(np.abs(gt_next_obs - curr_auto_preds)[:50, i], label="Curr Pred Error")
            plt.plot(np.abs(gt_next_obs - prev_auto_preds)[:50, i], label="Prev Pred Error")
            plt.legend()
            plt.savefig(f"{save_dir}/auto_error_dim_{i}.png")
            plt.close()

        # Plot total autoregressive error
        plt.figure(figsize=(10, 5))
        plt.plot(indices, prev_auto_err, label="SR from sim")
        plt.plot(indices, curr_auto_err, label="SR from sim + Residual MLP from real")
        plt.title("Total Auto-Regressive Error")
        plt.legend()
        plt.savefig(f"{save_dir}/auto_error_total.png")
        plt.close()

        # Plot predicted vs ground truth
        for i in range(gt_next_obs.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.plot(gt_next_obs[:50, i], label="GT")
            plt.plot(curr_auto_preds[:50, i], label="Curr Pred")
            plt.plot(prev_auto_preds[:50, i], label="Prev Pred")
            plt.legend()
            plt.savefig(f"{save_dir}/pred_{i}.png")
            plt.close()
            
    except Exception as e:
        print(f"Warning: Multi-step prediction failed: {e}")
        # Create basic plots with available data
        plt.figure(figsize=(10, 5))
        plt.plot(indices, one_step_err, label="One-step error")
        plt.title("One-step Prediction Error")
        plt.legend()
        plt.savefig(f"{save_dir}/one_step_error.png")
        plt.close() 