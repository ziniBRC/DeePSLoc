import os
from glob import glob
from PIL import Image

selected_labels = {'Nucleus': 0, 'Cytoplasm': 1, 'Vesicles': 2, 'Mitochondria': 3, 'Golgi Apparatus': 4, 'Endoplasmic Reticulum': 5}
cellDict = {
        # Nucleus
        'Nuclear membrane': 'Nucleus',

        'Nucleoli': 'Nucleus',
        'Nucleoli fibrillar center': 'Nucleus',

        'Nuclear bodies': 'Nucleus',
        'Nuclear speckles': 'Nucleus',
        'Nucleoplasm': 'Nucleus',
        'Nucleus': 'Nucleus',

        # Cytoplasm
        'actin filaments': 'Cytoplasm',
        'focal adhesion sites': 'Cytoplasm',

        'centrosome': 'Cytoplasm',
        'microtubule organizing center': 'Cytoplasm',

        'aggresome': 'Cytoplasm',
        'cytoplasmic bodies': 'Cytoplasm',
        'cytosol': 'Cytoplasm',
        'rods & rings': 'Cytoplasm',

        'intermediate filaments': 'Cytoplasm',

        'cleavage furrow': 'Cytoplasm',
        'cytokinetic bridge': 'Cytoplasm',
        'microtubule ends': 'Cytoplasm',
        'microtubules': 'Cytoplasm',
        'midbody': 'Cytoplasm',
        'midbody ring': 'Cytoplasm',
        'mitotic spindle': 'Cytoplasm',

        'mitochondria': 'Mitochondria',

        # Secretory
        'endoplasmic reticulum': 'Endoplasmic Reticulum',

        'golgi apparatus': 'Golgi Apparatus',

        'cell junctions': 'Plasma Membrane',
        'plasma membrane': 'Plasma Membrane',

        'secreted proteins': 'Secreted proteins',

        'endosomes': 'Vesicles',
        'lipid droplets': 'Vesicles',
        'lysosomes': 'Vesicles',
        'peroxisomes': 'Vesicles',
        'vesicles': 'Vesicles'
        }


def crop_data(org_data_dir='', name='', gene_label_dict={}, gene_count=[]):
    train_save_dir = 'data/train/cropHPA/'
    test_save_dir = 'data/test/cropHPA/'
    folders = os.listdir(org_data_dir)
    box = (400, 400, 2600, 2600)
    train_cnt = [1, 1, 1, 1, 1, 1]
    test_cnt = [1, 1, 1, 1, 1, 1]
    test_file_cnt = [0, 0, 0, 0, 0, 0]
    test_rate = 0.1

    if os.path.exists(train_save_dir) is False:
        os.makedirs(train_save_dir)
    for i in range(len(selected_labels)):
        if os.path.exists(train_save_dir + '/' + str(i)) is False:
            os.makedirs(train_save_dir + '/' + str(i))
    if os.path.exists(test_save_dir) is False:
        os.makedirs(test_save_dir)
    for i in range(len(selected_labels)):
        if os.path.exists(test_save_dir + '/' + str(i)) is False:
            os.makedirs(test_save_dir + '/' + str(i))
    for folder in folders:
        file_list = glob(org_data_dir + folder + '/*', )
        if len(file_list) == 0:
            continue
        if folder not in gene_label_dict:
            continue
        if gene_label_dict[folder] == -1:
            continue
        if test_file_cnt[gene_label_dict[folder]] < test_rate * gene_count[gene_label_dict[folder]]:
            save_dir = test_save_dir
            cnt = test_cnt
            test_file_cnt[gene_label_dict[folder]] += 1
            print(test_file_cnt, gene_count)
        else:
            save_dir = train_save_dir
            cnt = train_cnt
        for file in file_list:
            print(file)
            img = Image.open(file)
            cropped_img = img.crop(box)
            cropped_img.save(save_dir + str(gene_label_dict[folder]) + '/' + str(cnt[gene_label_dict[folder]]) + '.jpg')
            cnt[gene_label_dict[folder]] += 1


def get_gene_label_maps(org_data_dir='', labels_file_name=''):
    gene_label_dict = {}
    cnt = [0, 0, 0, 0, 0, 0]
    extra_cnt = 0
    false_cnt = 0
    label_cnt = 0
    with open(labels_file_name, 'r') as f:
        for line in f:
            gene_name = line.split('\t')[0]
            file_list = glob(org_data_dir + gene_name + '/*', )
            if len(file_list) == 0:
                continue

            labels = line.split('\t')[1][:-1]
            labels_split = labels.split(',')
            location_name = labels_split[0] if labels_split[0] not in cellDict else cellDict[labels_split[0]]
            label_cnt += len(labels_split)
            if len(labels_split) > 1:
                gene_label_dict[gene_name] = -1
                # for label in labels_split:
                #     location_name = label if label not in cellDict else cellDict[label]
                #     if location_name not in selected_labels:
                #         false_cnt += 1
                #     else:
                #         cnt[selected_labels[location_name]] += 1
            elif location_name not in selected_labels:
                gene_label_dict[gene_name] = -1
                false_cnt += 1
            else:
                gene_label_dict[gene_name] = selected_labels[location_name]
                cnt[selected_labels[location_name]] += 1
    return gene_label_dict, cnt


if __name__ == '__main__':
    gene_label_dict, gene_count = get_gene_label_maps(org_data_dir='/data/users/liuziyi/Data/hpa/', labels_file_name='enhanced_label.txt')
    crop_data('/data/users/liuziyi/Data/hpa/', gene_label_dict=gene_label_dict, gene_count=gene_count)
    pass