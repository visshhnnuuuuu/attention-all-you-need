from datasets import load_dataset


def load_translation_data(num_samples=50000):

    dataset = load_dataset(
        "cfilt/iitb-english-hindi",
        split="train"
    )

    # Shuffle the translation pairs
    dataset = dataset.shuffle(
        seed=42
    )

    # Select 50,000 random translation pairs
    dataset = dataset.select(
        range(
            min(num_samples, len(dataset))
        )
    )

    data = []

    for item in dataset:

        english = item["translation"]["en"].strip()
        hindi = item["translation"]["hi"].strip()

        if not english or not hindi:
            continue

        data.append({
            "src": hindi,
            "tgt": english
        })

    return data