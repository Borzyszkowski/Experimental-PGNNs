from pgnn import PhysicsGuidedNN


def main():
    physics_guided_nn = PhysicsGuidedNN(
        n_layers=3,
        n_nodes=12,
        n_epochs=10000,
        batch_size=1000,
        train_size=3000,
        lam=1000 * 0.05)

    physics_guided_nn.run()


if __name__ == '__main__':
    main()
