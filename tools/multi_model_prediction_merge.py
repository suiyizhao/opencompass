import json


def check_list_lengths_equal(dictionary):
    lengths = [len(value) for value in dictionary.values()]
    return len(set(lengths)) == 1


src_list = [
    'outputs/eval_internlm2_7b_pretrained_sf_data/20240620_140008/predictions/internlm2-chat-7b/SFData-val0612.json',  # noqa: E501
    'outputs/eval_internlm2_7b_pretrained_sf_data/20240620_140008/predictions/internlm2_7b_pretrained_full_sf-data-0611_e12_0612_hf/SFData-val0612.json',  # noqa: E501
    'outputs/eval_internlm2_7b_pretrained_sf_data/20240620_140008/predictions/internlm2_7b_pretrained_full_sf-data-0611_e12_0619_hf/SFData-val0612.json'  # noqa: E501
]
filename_map = {
    'internlm2-7b-chat': src_list[0],
    'internlm2-7b-1e': src_list[1],
    'internlm2-7b-4e': src_list[2]
}

dst_filename = 'outputs/final_results/20240620.jsonl'

dict_keys = list(filename_map.keys())

json_map = dict()
for key, value in filename_map.items():
    with open(value, 'r', encoding='utf-8') as file:
        # 读取文件内容并转换为 Python 对象
        json_map[key] = json.load(file)

assert check_list_lengths_equal(json_map)

dst_list = []
file_len = len(json_map[dict_keys[0]])

for i in range(file_len):
    result_dict = {}
    result_dict['origin_prompt'] = json_map[dict_keys[0]][str(
        i)]['origin_prompt']
    result_dict['gold'] = json_map[dict_keys[0]][str(i)]['gold']
    for key, value in json_map.items():
        result_dict[key] = json_map[key][str(i)]['prediction']
    dst_list.append(result_dict)

with open(dst_filename, 'w', encoding='utf-8') as f:
    for item in dst_list:
        json.dump(item, f, ensure_ascii=False, indent=4)
        f.write('\n')
