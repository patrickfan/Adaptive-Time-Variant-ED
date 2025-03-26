import tensorflow as tf
import numpy as np
from tqdm import tqdm

class Explainer():

    def attributions(self,
                     inputs,
                     baseline,
                     batch_size=50,
                     num_samples=100,
                     use_expectation=True,
                     output_indices=None,
                     verbose=False):

        raise Exception("Attributions have not been implemented " + \
                    "for this class. Likely, you have imported " + \
                    "the wrong class from this package.")
   

class PathExplainerTF(Explainer):

    def __init__(self, model, pass_original_input=False):

        self.model = model
        self.pass_original_input = pass_original_input
        self.eager_mode = False
        try:
            self.eager_mode = tf.executing_eagerly()
        except AttributeError:
            pass

    def accumulation_function(self,
                              batch_input,
                              batch_baseline,
                              batch_alphas,
                              output_index=None,
                              second_order=False,
                              interaction_index=None):

        if not second_order:
            batch_difference = batch_input - batch_baseline
            batch_interpolated = batch_alphas * batch_input + \
                                 (1.0 - batch_alphas) * batch_baseline

            with tf.GradientTape() as tape:
                tape.watch(batch_interpolated)

                if self.pass_original_input:
                    batch_predictions = self.model(batch_interpolated,
                                                   original_input=batch_input)
                else:
                    batch_predictions = self.model(batch_interpolated)

                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]
            batch_gradients = tape.gradient(batch_predictions, batch_interpolated)
            ########################

            batch_attributions = batch_gradients * batch_difference
            return batch_attributions


    def _sample_baseline(self, baseline, number_to_draw, use_expectation):

        if use_expectation:
            replace = baseline.shape[0] < number_to_draw
            sample_indices = np.random.choice(baseline.shape[0],
                                              size=number_to_draw,
                                              replace=replace)
            sampled_baseline = tf.gather(baseline, sample_indices)
        else:
            reps = np.ones(len(baseline.shape)).astype(int)
            reps[0] = number_to_draw
            sampled_baseline = np.tile(baseline, reps)
        return sampled_baseline

    def _sample_alphas(self, num_samples, use_expectation, use_product=False):

        if use_expectation:
            if use_product:
                alpha = np.random.uniform(low=0.0, high=1.0, size=num_samples).astype(np.float32)
                beta = np.random.uniform(low=0.0, high=1.0, size=num_samples).astype(np.float32)
                return alpha, beta
            else:
                return np.random.uniform(low=0.0, high=1.0, size=num_samples).astype(np.float32)
        else:
            if use_product:
                sqrt_samples = np.ceil(np.sqrt(num_samples)).astype(int)
                spaced_points = np.linspace(start=0.0,
                                            stop=1.0,
                                            num=sqrt_samples,
                                            endpoint=True).astype(np.float32)

                num_drawn = sqrt_samples * sqrt_samples
                slice_indices = np.round(np.linspace(start=0.0,
                                                     stop=num_drawn-1,
                                                     num=num_samples,
                                                     endpoint=True)).astype(int)

                ones_map = np.ones(sqrt_samples).astype(np.float32)
                beta = np.outer(spaced_points, ones_map).flatten()
                beta = beta[slice_indices]

                alpha = np.outer(ones_map, spaced_points).flatten()
                alpha = alpha[slice_indices]

                return alpha, beta
            else:
                return np.linspace(start=0.0,
                                   stop=1.0,
                                   num=num_samples,
                                   endpoint=True).astype(np.float32)

    def _single_attribution(self, current_input, current_baseline,
                            current_alphas, num_samples, batch_size,
                            use_expectation, output_index):

        current_input = np.expand_dims(current_input, axis=0)
        current_alphas = tf.reshape(current_alphas, (num_samples,) + \
                                    (1,) * (len(current_input.shape) - 1))

        attribution_array = []
        for j in range(0, num_samples, batch_size):
            number_to_draw = min(batch_size, num_samples - j)

            batch_baseline = self._sample_baseline(current_baseline,
                                                   number_to_draw,
                                                   use_expectation)
            batch_alphas = current_alphas[j:min(j + batch_size, num_samples)]

            reps = np.ones(len(current_input.shape)).astype(int)
            reps[0] = number_to_draw
            batch_input = tf.convert_to_tensor(np.tile(current_input, reps))

            batch_attributions = self.accumulation_function(batch_input,
                                                            batch_baseline,
                                                            batch_alphas,
                                                            output_index=output_index,
                                                            second_order=False,
                                                            interaction_index=None)
            attribution_array.append(batch_attributions)
        attribution_array = np.concatenate(attribution_array, axis=0)
        attributions = np.mean(attribution_array, axis=0)
        return attributions

    def _get_test_output(self,
                         inputs):
        return self.model(inputs[0:1])

    def _init_array(self,
                    inputs,
                    output_indices,
                    interaction_index=None,
                    as_interactions=False):

        test_output = self._get_test_output(inputs)
        is_multi_output = len(test_output.shape) > 1
        shape_tuple = inputs.shape
        num_classes = test_output.shape[-1]

        if as_interactions and interaction_index is None:
            shape_tuple = [inputs.shape[0], ] + \
                          2 * list(inputs.shape[1:])
            shape_tuple = tuple(shape_tuple)

        if is_multi_output and output_indices is None:
            num_classes = test_output.shape[-1]
            attributions = np.zeros((num_classes,) + shape_tuple)
        elif not is_multi_output and output_indices is not None:
            raise ValueError('Provided output_indices but ' + \
                             'model is not multi output!')
        else:
            attributions = np.zeros(shape_tuple)

        return attributions, is_multi_output, num_classes

    def attributions(self, inputs, baseline,
                     batch_size=50, num_samples=100,
                     use_expectation=True, output_indices=None,
                     verbose=False):

        attributions, is_multi_output, num_classes = self._init_array(inputs,
                                                                      output_indices)

        input_iterable = enumerate(inputs)
        if verbose:
            input_iterable = enumerate(tqdm(inputs))

        for i, current_input in input_iterable:
            current_alphas = self._sample_alphas(num_samples, use_expectation)

            if not use_expectation and baseline.shape[0] > 1:
                current_baseline = np.expand_dims(baseline[i], axis=0)
            else:
                current_baseline = baseline

            if is_multi_output:
                if output_indices is not None:
                    if isinstance(output_indices, int):
                        output_index = output_indices
                    else:
                        output_index = output_indices[i]
                    current_attributions = self._single_attribution(current_input,
                                                                    current_baseline,
                                                                    current_alphas,
                                                                    num_samples,
                                                                    batch_size,
                                                                    use_expectation,
                                                                    output_index)
                    attributions[i] = current_attributions
                else:
                    for output_index in range(num_classes):
                        current_attributions = self._single_attribution(current_input,
                                                                        current_baseline,
                                                                        current_alphas,
                                                                        num_samples,
                                                                        batch_size,
                                                                        use_expectation,
                                                                        output_index)
                        attributions[output_index, i] = current_attributions
            else:
                current_attributions = self._single_attribution(current_input,
                                                                current_baseline,
                                                                current_alphas,
                                                                num_samples,
                                                                batch_size,
                                                                use_expectation,
                                                                None)
                attributions[i] = current_attributions
        return attributions

