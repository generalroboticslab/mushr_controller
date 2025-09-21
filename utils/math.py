import jax
import jax.numpy as jnp

@jax.jit
def quat2euler(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw)."""
    
    # Roll
    roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # Pitch
    sinp = 2 * (qw * qy - qz * qx)
    pitch = jnp.where(jnp.abs(sinp) >= 1, jnp.sign(sinp) * (jnp.pi / 2), jnp.arcsin(sinp))

    # Yaw
    yaw = jnp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    return jnp.array([roll, pitch, yaw])