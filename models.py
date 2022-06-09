import time
import random
import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self, conv_output_size=7, image_channels=1):
        super(Model1, self).__init__()

        # Convolutional network 1.
        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolutional network 2.
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer where each node represents a class.
        self.out = nn.Linear(
            in_features=conv_output_size * conv_output_size * 32,  # The activation volume of the last convolutional network.
            out_features=10,  # The number of output classes (digits from 0 to 9).
        )

    def forward(self, x):
        # Pass the input image through the first convolutional network.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Pass the activation volume through the second convolutional network.
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Pass the activation volume through the final, fully connected layer.
        x = x.view(x.size(0), -1)  # Flatten the activation volume to a 1D vector with 7 * 7 * 32 elements.
        output = self.out(x)

        return output

    def train_model(self, data_loaders, num_epochs, optimizer, loss_function=nn.CrossEntropyLoss()):
        # Initialize logging lists for losses.
        # Epoch level.
        logs_xs_epochs = []
        logs_ys_losses_epochs = []
        logs_ys_durations_epochs = []
        logs_xs_eval_epochs = []
        logs_ys_eval_losses_epochs = []
        logs_ys_eval_accuracy_epochs = []

        # If we have a GPU, make sure we train on the GPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()  # Set the model in train mode.

        num_batches = len(data_loaders["train"])

        for i_epoch in range(num_epochs):
            xs, ys = None, None
            for i_batch, (xs_batch, ys_batch) in enumerate(data_loaders["train"]):
                assert i_batch == 0, f"Found more than 1 batch per epoch!"
                # xs_batch = a batch of images.
                # ys_batch = a batch of labels.
                xs, ys = xs_batch, ys_batch

            # Send the current batch to the same device where the model resides.
            xs = xs.to(device)
            ys = ys.to(device)

            curr_epoch_time_start = time.time()

            # Reset the gradients for this batch.
            optimizer.zero_grad()

            # Run the batch of samples through the network.
            ys_predicted = self(xs)
            # Compute the loss between the predictions and the ground truth labels.
            loss = loss_function(ys_predicted, ys)
            # Back propagate the loss to train the model.
            loss.backward()
            # Optimize (update weights based on the newly computed gradients).
            optimizer.step()

            curr_epoch_time_end = time.time()
            curr_epoch_duration = curr_epoch_time_end - curr_epoch_time_start

            test_loss, test_accuracy = self.evaluate(data_loaders, loss_function)

            logs_xs_epochs.append(i_epoch + 1)
            logs_ys_losses_epochs.append(loss.item())
            logs_ys_durations_epochs.append(curr_epoch_duration)
            logs_xs_eval_epochs.append(i_epoch + 1)
            logs_ys_eval_losses_epochs.append(test_loss)
            logs_ys_eval_accuracy_epochs.append(test_accuracy)

            print(f"Epoch {i_epoch + 1}/{num_epochs} train loss: {loss.item():.3f}")
            print(f"Epoch {i_epoch + 1}/{num_epochs} test loss: {test_loss:.3f} | test accuracy: {test_accuracy * 100:.3f}")

        return (
            logs_xs_epochs, logs_ys_losses_epochs, logs_ys_durations_epochs,
            logs_xs_eval_epochs, logs_ys_eval_losses_epochs, logs_ys_eval_accuracy_epochs
        )

    @torch.no_grad()
    def evaluate(self, data_loaders, loss_function=nn.CrossEntropyLoss()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_correct = 0
        num_all = 0

        running_loss_eval = 0.0

        for xs_batch, ys_batch in data_loaders["test"]:
            xs_batch = xs_batch.to(device)
            ys_batch = ys_batch.to(device)

            ys_raw_predictions = self(xs_batch)
            _, ys_predictions = torch.max(ys_raw_predictions, 1)

            num_all += ys_batch.size(0)
            num_correct += (ys_predictions == ys_batch).sum().item()

            loss = loss_function(ys_raw_predictions, ys_batch)
            running_loss_eval += loss.item()

        accuracy = num_correct / num_all

        return running_loss_eval, accuracy


class PSOModel(nn.Module):
    def __init__(
        self,
        model_class,
        model_params,
        num_particles=25,
        v_max=1.0,
        search_space=(-50.0, 50.0),
        inertia_weight=0.8,
        cognitive_coeff=0.1,
        social_coeff=0.1,
    ):
        super(PSOModel, self).__init__()

        if search_space[0] > search_space[1]:
            raise ValueError(
                f"The search space must be provided as (min_value, max_value) where min_value <= max_value."
            )
        if v_max <= 0:
            raise ValueError(
                f"The maximum absolute velocity must be a positive integer."
            )

        self.num_particles = num_particles
        self.v_max = v_max
        self.search_space = search_space
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff

        self.model = model_class(**model_params)

        self.num_variables = 0
        self.param_flattened_intervals = []
        for i_param, param in enumerate(self.model.parameters()):
            num_variables_param = torch.flatten(param).shape[0]
            # print(f"Parameter {i_param} has: {num_variables_param} variables (shape {param.size()}).")
            self.param_flattened_intervals.append(
                (self.num_variables, self.num_variables + num_variables_param - 1)
            )
            self.num_variables += num_variables_param

        # print(f"In total, there are {self.num_variables} 1D variables that must be optimized.")
        # print()

        # Initialize the particles with random positions from the search space.
        self.particles_x = (
            torch.ones(size=(self.num_particles, self.num_variables), dtype=torch.float) * self.search_space[0] +
            torch.rand(size=(self.num_particles, self.num_variables), dtype=torch.float) *
            (search_space[1] - search_space[0])
        )
        # Initialize the particle velocities with random values for each dimension from [-v_max, v_max).
        self.particles_v = (
            torch.ones(size=(self.num_particles, self.num_variables), dtype=torch.float) * -self.v_max +
            torch.rand(size=(self.num_particles, self.num_variables), dtype=torch.float) * 2 * self.v_max
        )

        self.particles_personal_best = torch.zeros(size=(self.num_particles, self.num_variables), dtype=torch.float)
        self.particles_personal_best_loss = torch.ones(size=(self.num_particles,), dtype=torch.float) * -1.0

        self.particles_global_best_i = random.randint(0, self.num_particles - 1)
        self.particles_global_best = self.particles_x[self.particles_global_best_i]
        self.particles_global_best_loss = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move everything to GPU, if available.
        self.particles_x.to(device)
        self.particles_v.to(device)
        self.particles_personal_best.to(device)
        self.particles_personal_best_loss.to(device)

    @torch.no_grad()
    def train_model(self, data_loaders, num_epochs, optimizer, loss_function=nn.CrossEntropyLoss()):
        # Initialize logging lists for losses.
        # Epoch level.
        logs_xs_epochs = []
        logs_ys_losses_epochs = []  # Best training loss at each epoch from all particles.
        logs_ys_durations_epochs = []

        logs_xs_eval_epochs = []
        logs_ys_eval_losses_epochs = []
        logs_ys_eval_accuracy_epochs = []

        # If we have a GPU, make sure we train on the GPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.model.to(device)
        self.train()  # Set the model in train mode.

        num_batches = len(data_loaders["train"])
        if num_batches != 1:
            raise ValueError(
                f"Since we're dealing with PSO, we should work with the entire training set when computing the "
                f"'fitness' as the loss function, so no batches should be used, i.e num_batches must be 1."
            )

        # First, find the initial global best and set the personal bests if this is the first time we run train_model.
        # Basically, run an extra, initial epoch, but don't update the particles positions / velocities.
        if self.particles_global_best_loss is None:
            xs, ys = None, None
            for i_batch, (xs_batch, ys_batch) in enumerate(data_loaders["train"]):
                assert i_batch == 0, f"Found more than 1 batch per epoch!"
                xs, ys = xs_batch, ys_batch

            # Send the current batch to the same device where the model resides.
            xs = xs.to(device)
            ys = ys.to(device)

            for i_particle in range(self.num_particles):
                # self.update_particle(i_particle)  # Don't because we just evaluate the current state of the perticle.
                self.set_model_to_particle(self.particles_x[i_particle])
                # Run the samples through the network.
                ys_predicted = self.model(xs)
                # Compute the loss between the predictions and the ground truth labels.
                loss = loss_function(ys_predicted, ys)

                curr_personal_best_loss = self.particles_personal_best_loss[i_particle].item()
                if curr_personal_best_loss < 0 or loss.item() < curr_personal_best_loss:
                    self.particles_personal_best_loss[i_particle] = loss
                    self.particles_personal_best[i_particle] = self.particles_x[i_particle]

                    if self.particles_global_best_loss is None or loss.item() < self.particles_global_best_loss:
                        self.particles_global_best = self.particles_x[i_particle]
                        self.particles_global_best_i = i_particle
                        self.particles_global_best_loss = loss

        # Now do the actual search.
        for i_epoch in range(num_epochs):
            xs, ys = None, None
            for i_batch, (xs_batch, ys_batch) in enumerate(data_loaders["train"]):
                assert i_batch == 0, f"Found more than 1 batch per epoch!"
                # xs_batch = a batch of images.
                # ys_batch = a batch of labels.
                xs, ys = xs_batch, ys_batch

            # Send the current batch to the same device where the model resides.
            xs = xs.to(device)
            ys = ys.to(device)

            best_curr_epoch_loss = None
            best_curr_epoch_particle = None
            best_curr_epoch_particle_i = None
            curr_epoch_time_start = time.time()
            for i_particle in range(self.num_particles):
                self.update_particle(i_particle)
                self.set_model_to_particle(self.particles_x[i_particle])
                # Run the samples through the network.
                ys_predicted = self.model(xs)
                # Compute the loss between the predictions and the ground truth labels.
                loss = loss_function(ys_predicted, ys)

                if best_curr_epoch_loss is None or loss < best_curr_epoch_loss:
                    best_curr_epoch_loss = loss
                    best_curr_epoch_particle = self.particles_x[i_particle]
                    best_curr_epoch_particle_i = i_particle

                curr_personal_best_loss = self.particles_personal_best_loss[i_particle].item()
                if curr_personal_best_loss < 0 or loss.item() < curr_personal_best_loss:
                    self.particles_personal_best_loss[i_particle] = loss
                    self.particles_personal_best[i_particle] = self.particles_x[i_particle]

                    if self.particles_global_best_loss is None or loss.item() < self.particles_global_best_loss:
                        self.particles_global_best = self.particles_x[i_particle]
                        self.particles_global_best_i = i_particle
                        self.particles_global_best_loss = loss

            curr_epoch_time_end = time.time()
            curr_epoch_duration = curr_epoch_time_end - curr_epoch_time_start

            # Evaluate the best particle this epoch on the test set.
            print(
                f"Epoch {i_epoch + 1}/{num_epochs} best train loss: {best_curr_epoch_loss:.3f} "
                f"by particle {best_curr_epoch_particle_i}. | "
                f"Global best train loss: {self.particles_global_best_loss:.3f}"
            )
            self.set_model_to_particle(best_curr_epoch_particle)
            test_loss, test_accuracy = self.evaluate(data_loaders, loss_function)
            print(
                f"Epoch {i_epoch + 1}/{num_epochs} test_loss: {test_loss:.3f} | test_accuracy: {test_accuracy * 100:.3f} | "
                f"Particle {best_curr_epoch_particle_i}."
            )
            print(f"Epoch {i_epoch + 1}/{num_epochs} duration: {curr_epoch_duration:.2f}s")
            print()

            logs_xs_epochs.append(i_epoch + 1)
            logs_ys_losses_epochs.append(best_curr_epoch_loss)
            logs_ys_durations_epochs.append(curr_epoch_duration)
            logs_xs_eval_epochs.append(i_epoch + 1)
            logs_ys_eval_losses_epochs.append(test_loss)
            logs_ys_eval_accuracy_epochs.append(test_accuracy)

        return (
            logs_xs_epochs, logs_ys_losses_epochs, logs_ys_durations_epochs,
            logs_xs_eval_epochs, logs_ys_eval_losses_epochs, logs_ys_eval_accuracy_epochs
        )

    def update_particle(self, i_particle):
        particle_x_old = self.particles_x[i_particle]
        particle_v_old = self.particles_v[i_particle]
        particle_personal_best = self.particles_personal_best[i_particle]
        global_best = self.particles_global_best

        # Randomly pick the current cognitive and social random values.
        cognitive_rand = torch.rand(self.num_variables, dtype=torch.float)
        social_rand = torch.rand(self.num_variables, dtype=torch.float)

        particle_v_new = (
            self.inertia_weight * particle_v_old +
            self.cognitive_coeff * cognitive_rand * (particle_personal_best - particle_x_old) +
            self.social_coeff * social_rand * (global_best - particle_x_old)
        )
        particle_v_new = torch.clamp(particle_v_new, min=-self.v_max, max=self.v_max)

        particle_x_new = particle_x_old + particle_v_new
        particle_x_new = torch.clamp(particle_x_new, min=self.search_space[0], max=self.search_space[1])

        self.particles_v[i_particle] = particle_v_new
        self.particles_x[i_particle] = particle_x_new

    def set_model_to_particle(self, particle):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i_param, param in enumerate(self.model.parameters()):
            param_flattened_interval = self.param_flattened_intervals[i_param]
            param_flattened_start, param_flattened_end = param_flattened_interval
            param_flattened_variables = particle[param_flattened_start:(param_flattened_end + 1)]

            param.data = param_flattened_variables.view(*param.shape).to(device)

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def evaluate(self, data_loaders, loss_function=nn.CrossEntropyLoss()):
        return self.model.evaluate(data_loaders, loss_function)


if __name__ == "__main__":
    pso_model = PSOModel(
        model_class=Model1,
        model_params={
            "conv_output_size": 7,
            "image_channels": 1,
        },
        num_particles=25,
        v_max=1.0,
        search_space=(-50.0, 50.0),
        inertia_weight=0.8,
        cognitive_coeff=0.1,
        social_coeff=0.1,
    )
