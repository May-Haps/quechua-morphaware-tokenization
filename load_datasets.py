from datasets import load_dataset

quechua_only_dataset_id = "Llamacha/monolingual-quechua-iic"
quechua_spanish_dataset_id = "somosnlp-hackathon-2022/spanish-to-quechua"

q_dataset = load_dataset(quechua_only_dataset_id)
qs_dataset = load_dataset(quechua_spanish_dataset_id)

# print(q_dataset)
# print(qs_dataset)
#
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 175408
#     })
# })
# DatasetDict({
#     train: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 102747
#     })
#     validation: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 12844
#     })
#     test: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 12843
#     })
# })
