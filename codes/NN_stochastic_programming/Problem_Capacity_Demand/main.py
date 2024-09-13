# main.py

import data_generation
import neural_network
import optimization

def main():
    # Step 1: Generate training data
    print("Generating training data...")
    first_stage_decisions = np.linspace(50, 150, 20)
    num_scenarios = 1000
    training_data = data_generation.generate_training_data(first_stage_decisions, num_scenarios)
    training_data.to_csv('training_data.csv', index=False)

    # Step 2: Train the approximation function
    print("Training the approximation function...")
    neural_network.train_approximation_function('training_data.csv')

    # Step 3: Solve the optimization model
    print("Solving the optimization model...")
    optimization.define_optimization_model()

if __name__ == "__main__":
    main()
