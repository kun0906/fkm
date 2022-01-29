"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 collect_results.py
"""
# Email: kun.bj@outlook.com
import collections
import copy
import json
import os
import traceback

import numpy as np
import xlsxwriter

from fkm.experiment_cases import get_experiment_params
from fkm.kun import gen_cases
from fkm.utils.utils_func import load


def save2xls(worksheet, py_name, case, column_idx, client_epochs):
    dataset = case['dataset']
    data_details = case['data_details']
    algorithm = case['algorithm']
    params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm, client_epochs=client_epochs,
                                   p3=py_name)
    # pprint(params)
    out_dir = params['out_dir']
    print(f'out_dir: {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # set background color
    cell_format = workbook.add_format()
    cell_format.set_pattern(1)  # This is optional when using a solid fill.
    cell_format.set_text_wrap()
    # new cell_format to add more formats to one cell
    cell_format2 = copy.deepcopy(cell_format)
    cell_format2.set_bg_color('FF0000')
    # cell_format.set_bg_color('#FFFFFE')

    row = 0
    # add dataset details, e.g., plot
    worksheet.set_row(row, 100)  # set row height to 100
    scale = 0.248
    dataset_img = os.path.join(out_dir, params['p1']+'.png')
    worksheet.insert_image(row, column_idx, dataset_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
    row += 1

    # Widen the first column to make the text clearer.
    worksheet.set_column(column_idx, column_idx, width=50, cell_format=cell_format)
    # Insert an image offset in the cell.
    if column_idx == 0:
        s = f'{out_dir}'
    else:
        s = ''
    worksheet.write(row, column_idx, s)
    row += 1

    # write the second row
    s = f'{py_name}.py'
    if column_idx < 3:  # centralized kmeans
        cell_format.set_bg_color('#F0FEFE')  # #RRGGBB
        # #merge_range(first_row, first_col, last_row, last_col, data[, cell_format])
        # cf1 = copy.deepcopy(cell_format)
        # cf1.set_align('center')
        # cf1.set_bold(bold=True)
        # tmp = worksheet.merge_range(row, column_idx+1, row, column_idx+2, s)  # not work
    elif 3 <= column_idx < 5:  # Server random/true
        cell_format.set_bg_color('#FEF0FE')
        # worksheet.merge_range(row, column_idx, row, column_idx + 1, s, cell_format)
    elif 5 <= column_idx < 8:  # Server average
        cell_format.set_bg_color('#FEFEF0')
        # worksheet.merge_range(row, column_idx, row, column_idx+2, s, cell_format)
    elif 8 <= column_idx < 11:  # Server greedy
        cell_format.set_bg_color('#F0FBFB')
        # worksheet.merge_range(row, column_idx, row, column_idx+2, s, cell_format)
    elif 11 <= column_idx < 14:  # Server center
        cell_format.set_bg_color('#F0FBF0')
        # worksheet.merge_range(row, column_idx, row, column_idx+2, s, cell_format)
    elif 14 <= column_idx < 17:  # Server 2k
        cell_format.set_bg_color('#F0F0FB')
        # worksheet.merge_range(row, column_idx, row, column_idx + 2, s, cell_format)
    elif 17 <= column_idx < 20:  # Server k_k
        cell_format.set_bg_color('#F0F0F0')
        # worksheet.merge_range(row, column_idx, row, column_idx+2, s, cell_format)
    else:
        cell_format.set_bg_color('#FFFFFF')  # white
    worksheet.write(row, column_idx, s)
    row += 1
    # Insert an image offset in the cell.
    s = f'{dataset}\n{data_details}\n{algorithm}'
    worksheet.set_row(row, 50)  # set row height to 100
    worksheet.write(row, column_idx, s)
    row += 1
    try:
        # read scores from out.txt
        server_init_centroids = params['server_init_centroids']
        client_init_centroids = params['client_init_centroids']
        if False:
            print('deprecated')
            out_txt = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
                                            f'Client_{client_init_centroids}.txt')
            with open(out_txt, 'r') as f:
                data = json.load(f)
            s = ' '
            for k, vs in data.items():
                for split in vs.keys():
                    s += f'{split}:\n'
                    for metric, score in vs[split].items():
                        s += f'\t{metric}: ' + '+/-'.join(f'{v:.2f}' for v in score) + '\n'
                    s += '\n'
        else:
            out_dat = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
                                            f'Client_{client_init_centroids}-histories.dat')
            histories = load(out_dat)
            for n_clients, history_res in histories.items():
                results_avg = history_res['results_avg']
                n_clients = history_res['n_clients']
                results = history_res['history']['results']
                s = ''
                training_iterations_lst = []
                scores_lst = []
                final_centroids_lst = []
                for vs in results:
                    seed = vs['seed']
                    # print(f'seed: {seed}')
                    training_iterations_lst.append(vs['training_iterations'])
                    scores_lst.append(vs['scores'])
                    final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in cen) + ')' for cen
                                            in vs['final_centroids'].tolist()]

                for split in scores_lst[0].keys():  # ['train', 'test']
                    s += f'{split}:\n'
                    if split == 'train':
                        s += f'\titerations: {np.mean(training_iterations_lst):.2f} +/- ' \
                             f'{np.std(training_iterations_lst):.2f}\n'
                    else:
                        s += 'Iterations: \n'
                    for metric in scores_lst[0][split].keys():
                        metric_scores = [scores[split][metric] for scores in scores_lst]
                        s += f'\t{metric}: {np.mean(metric_scores):.2f} +/- {np.std(metric_scores):.2f}\n'
                    s += '\n'

                # final centroids distribution
                s += 'final centroids distribution: \n'
                ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
                tot_centroids = len(final_centroids_lst)
                s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
                                 cen_, cnt_ in ss_)

                data = s
                # Insert an image offset in the cell.
                worksheet.set_row(row, 400)  # set row height to 100
                # cell_format = workbook.add_format({'bold': True, 'italic': True})
                # cell_format2 = workbook.add_format()
                cell_format.set_align('top')
                worksheet.write(row, column_idx, data, cell_format)
                break
    except Exception as e:
        print(f'Error: {e}')
        data = '-'
    row += 1

    # worksheet.write('A12', 'Insert an image with an offset:')
    n_clients = 0 if 'Centralized' in algorithm else params['n_clients']
    sub_dir = f'Clients_{n_clients}'
    if 'FEMNIST' in params['p0']:
        centroids_img = os.path.join(out_dir, sub_dir, 'over_time', f'M={n_clients}-centroids.png')
    else:
        centroids_img = os.path.join(out_dir, sub_dir, f'M={n_clients}-Centroids.png')
    print(f'{centroids_img} exist: {os.path.exists(centroids_img)}')
    worksheet.set_row(row, 300)  # set row height to 30
    scale = 0.248
    worksheet.insert_image(row, column_idx, centroids_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
    row += 1

    score_img = os.path.join(out_dir, sub_dir, 'over_time', f'M={n_clients}-scores.png')
    if not os.path.exists(score_img):
        score_img = os.path.join(out_dir, sub_dir, 'over_time', f'M={n_clients}.png')
    print(f'{score_img} exist: {os.path.exists(score_img)}')
    worksheet.set_row(row, 300)  # set row height to 30
    worksheet.insert_image(row, column_idx, score_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
    row += 1


if __name__ == '__main__':
    tot_cnt = 0
    client_epochs = 1
    for dataset in ['3GAUSSIANS','2GAUSSIANS', '5GAUSSIANS']:  # [ '2GAUSSIANS', 'FEMNIST']:
        # dataset = 'FEMNIST'
        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(f'{dataset}-Client_epochs_{client_epochs}.xlsx')

        py_names = [
            'Centralized_Kmeans',
            'Stanford_random_initialization',
            'Stanford_average_initialization',
            'Our_greedy_initialization',
            'Our_greedy_center',
            'Our_greedy_2K',
            'Our_greedy_K_K',
        ]
        cnt = 0
        if dataset == 'FEMNIST':
            data_details_lst = ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
                                '1client_multiwriters_1digit']
        elif dataset == '2GAUSSIANS':
            data_details_lst = [
                '1client_1cluster', '1client_0.7cluster1_0.3cluster2',
                '1client_ylt0', '1client_xlt0',
                '1client_1cluster_diff_sigma', '1client_1cluster_diff_sigma_n',
                '1client_xlt0_2',
            ]
        elif dataset == '3GAUSSIANS':
            data_details_lst = [
                '1client_1cluster', '1client_0.7cluster1_0.3cluster2',
                '1client_ylt0', '1client_xlt0',
                '1client_1cluster_diff_sigma', '1client_1cluster_diff_sigma_n',
                '1client_xlt0_2',
            ]
        elif dataset == '5GAUSSIANS':
            data_details_lst = [
                '5clients_5clusters', '5clients_4clusters', '5clients_3clusters',
            ]
        else:
            msg = f'{dataset}'
            raise NotImplementedError(msg)
        for data_details in data_details_lst:
            cnt_ = 0
            print(f'xlsx_sheet_name: {data_details}')
            worksheet = workbook.add_worksheet(name=data_details[:30])
            cnt_ = 0
            for py_name in py_names:
                for case in gen_cases(py_name, dataset, data_details):
                    try:
                        save2xls(worksheet, py_name, case, cnt_, client_epochs)
                    except Exception as e:
                        traceback.print_exc()
                    cnt_ += 1
            print(f'{py_name}: {cnt_} cases.\n')
            cnt += cnt_
        tot_cnt += cnt
        print(f'* {dataset} cases: {cnt}\n')

        workbook.close()
        print()

    print(f'*** Total cases: {tot_cnt}')
