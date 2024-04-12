"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import wandb

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


utils.setup_logger()


def main(args):

    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric(args.metric)

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature)
                embedding = embedding.cpu() if embedding is not None else None

                # Only compute accuracy if question is answerable.
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})

                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
