import retro
import pygame
import os

# Creates a Retro environment. "microgame_name" should be a directory inside the integration
# folder that contains a state file of the same name.
def create_env(microgame_name="CrazyCars"):
    # Ensure BIOS file exists to launch GBA games.
    bios_path = "/home/mondolo/stable-retro/retro/cores/system/gba_bios.bin"
    print("BIOS exists:", os.path.exists(bios_path))

    # Look inside the contrib folder and print path to the ROM file (if it exists).
    print("ROM path:", retro.data.get_file_path(
        "WarioWareIncMegaMicrogames-GbAdvance",
        "rom.gba",
        retro.data.Integrations.CONTRIB
    ), "\n")

    # Verify the ROM checksum to confirm it matches the expected dataset hash.
    retro.data.verify_hash("WarioWareIncMegaMicrogames-GbAdvance", inttype=retro.data.Integrations.CONTRIB)

    # "base_path" should reflect where the state file is found.
    base_path = "/home/mondolo/stable-retro/retro/data/contrib/WarioWareIncMegaMicrogames-GbAdvance"
    state_path = os.path.join(base_path, f"{microgame_name}/{microgame_name}.state")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Microgame state not found: {state_path}")

    print(f"Loading microgame: {microgame_name}\n")

    microgame_env = retro.make(
        game="WarioWareIncMegaMicrogames-GbAdvance",
        state=state_path,
        inttype=retro.data.Integrations.CONTRIB,
        render_mode="human"
    )

    # Print game system, observation space info, and action space.
    print("System:", microgame_env.system)
    print("Observation shape:", microgame_env.observation_space.shape)
    print("Action space:", microgame_env.action_space.n)
    print("Buttons:", microgame_env.buttons)

    return microgame_env

# Create gym environment with a default state file of "Crazy Cars".
env = create_env()

# Initialize pygame for keyboard input.
pygame.init()
window = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Retro Controller")

# Map keys to GBA buttons.
key_map = {
    pygame.K_z: "B",
    pygame.K_x: "A",
    pygame.K_a: "L",
    pygame.K_s: "R",
    pygame.K_SPACE: "START",
    pygame.K_BACKSPACE: "SELECT",
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
}

# Build lookup index for quick button -> position.
button_index = {b: i for i, b in enumerate(env.buttons)}

obs = env.reset()
running = True
clock = pygame.time.Clock()

# Run environment until user closes the Retro Controller window.
while running:
    action = [0] * len(env.buttons)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    for key, button in key_map.items():
        if keys[key]:
            action[button_index[button]] = 1

    obs, rew, done, trunc, info = env.step(action)
    env.render()
    clock.tick(60) # 60 FPS

env.close()
pygame.quit()
