import taichi as ti

ti.init(arch=ti.cpu)

max_num_particles = 1024

x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)

particle_mass = 1.0
dt = 1e-3
substeps = 10

fixed = ti.field(dtype=ti.i32, shape=max_num_particles)
spring_Y = ti.field(dtype=ti.f32, shape=())
paused = ti.field(dtype=ti.i32, shape=())
num_particles = ti.field(dtype=ti.i32, shape=())
rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles, max_num_particles))
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def substep():
    n = num_particles[None]

    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] -
                                           1) * d

                # Dashpot damping
                v_rel = (v[i] - v[j]).dot(d)
                f[i] += -dashpot_damping[None] * v_rel * d

    # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

        # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further


@ti.kernel
def add_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    new_particle_id = num_particles[None]
    v[new_particle_id] = [0, 0]
    x[new_particle_id] = ti.Vector([pos_x, pos_y])
    fixed[new_particle_id] = fixed_

    for i in range(num_particles[None]):
        if (x[new_particle_id] - x[i]).norm() < 0.15:
            rest_length[new_particle_id, i] = 0.1
            rest_length[i, new_particle_id] = 0.1

    num_particles[None] += 1

@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 100



def main():
    gui = ti.GUI('Explicit Mass Spring System', background_color=0xDDDDDD)

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100

    # add_particle(0.3, 0.3, True)
    # add_particle(0.3, 0.4, False)
    # add_particle(0.4, 0.4, False)

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                add_particle(e.pos[0], e.pos[1], int(gui.is_pressed(ti.GUI.SHIFT)))
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]
            elif e.key == 'c':
                num_particles[None] = 0
                rest_length.fill(0)
            elif e.key == 'y':
                if gui.is_pressed('Shift'):
                    spring_Y[None] /= 1.1
                else:
                    spring_Y[None] *= 1.1
            elif e.key == 'd':
                if gui.is_pressed('Shift'):
                    drag_damping[None] /= 1.1
                else:
                    drag_damping[None] *= 1.1
            elif e.key == 'x':
                if gui.is_pressed('Shift'):
                    dashpot_damping[None] /= 1.1
                else:
                    dashpot_damping[None] *= 1.1
            elif e.key == 'd':
                if gui.is_pressed('Shift'):
                    drag_damping[None] /= 1.1
                else:
                    drag_damping[None] *= 1.1
            elif e.key == 'x':
                if gui.is_pressed('Shift'):
                    dashpot_damping[None] /= 1.1
                else:
                    dashpot_damping[None] *= 1.1

        if not paused[None]:
            for step in range(substeps):
                substep()

        if gui.is_pressed(ti.GUI.RMB):
            cursor_pos = gui.get_cursor_pos()
            attract(cursor_pos[0], cursor_pos[1])

        X = x.to_numpy()
        n = num_particles[None]
        for i in range(n):
            for j in range(n):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x888888)

        for i in range(n):
            c = 0xFF0000 if fixed[i] else 0x0
            gui.circle(X[i], color=c, radius=5)

        gui.text(
            content=
            f'Left click: add mass point (with shift to fix); Right click: attract',
            pos=(0, 0.99),
            color=0x0)
        gui.text(content=f'C: clear all; Space: pause',
                 pos=(0, 0.95),
                 color=0x0)
        gui.text(content=f'Y: Spring Young\'s modulus {spring_Y[None]:.1f}',
                 pos=(0, 0.9),
                 color=0x0)
        gui.text(content=f'D: Drag damping {drag_damping[None]:.2f}',
                 pos=(0, 0.85),
                 color=0x0)
        gui.text(content=f'X: Dashpot damping {dashpot_damping[None]:.2f}',
                 pos=(0, 0.8),
                 color=0x0)

        gui.show()


if __name__ == '__main__':
    main()
