import racecarGymEnv as e
import gym
from stable_baselines3 import PPO
import os


env = e.RacecarGymEnv(renders=False)
model_path = "racecar_ppo_mio.zip"

# Intentar cargar el modelo si existe
if os.path.exists(model_path):
    print("Cargando el modelo previamente entrenado...")
    model = PPO.load(model_path, env=env)  # Cargar el modelo
else:
    print("Modelo no encontrado. Creando un nuevo modelo...")
    model = PPO("MlpPolicy", env, verbose=1)  # Crear un nuevo modelo
# Entrenar el agente
model.learn(total_timesteps=30000)


# Guardar el modelo entrenado
model.save("racecar_ppo_mio")
env.close()


# # Probar el modelo entrenado
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()  # Mostrar el entorno
#     if done:
#         obs = env.reset()

# # Cerrar el entorno
# env.close()
