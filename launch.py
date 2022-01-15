from asyncore import write
from random_forest import *
from data_load import *

from random_forest import *
from data_load import *

def main():

	data_absence = read_file("datasets_v2/Absenteeism_at_work.csv")
	dataset_absence = load_data(data_absence, 3)
	dataset_absence = dataset_absence[1:]
	str2float(dataset_absence)

	data_car = read_file("datasets_v2/car.data")
	dataset_car = load_data(data_car, 1)
	str2int(dataset_car)
    
	data_opt = read_file("datasets_v2/batch_optdigits.tra")
	dataset_opt = load_data(data_opt, 1)
	str2float(dataset_opt)

	data_gas = read_file("datasets_v2/GasSensorDataset/batch1.dat")
	dataset_gas = load_data(data_gas, 2)
	dataset_gas = process_dat_data(dataset_gas)
	str2float(dataset_gas)
	
	sk_scores = []
	for_scores = []
	col_1 = []
	col_2 = []
	col_3 = []
	col_4 = []
	col_5 = []
	col_6 = []
	col_7 = []
	col_8 = []


	for dataset in [dataset_car, dataset_absence, dataset_gas]:
		sample_size = 0.05
		folds_num = 5
		for threshold in [0.2 , 0.5, 0.9]:
			print('Dla mojej informacji - threshold: ', threshold, '\n')
			for trees_num in range(1,10):
				sk_scores.clear()
				for_scores.clear()
				for i in [1,2,3,5]:
					seed(i)
					scores = evaluate_algorithm(dataset, folds_num, trees_num, sample_size, threshold)
					score = sum(scores[0])/float(len(scores[0]))
					for_scores.append(score)
					sk_score = sum(scores[1])/float(len(scores[1]))
					sk_scores.append(sk_score)
				if dataset == dataset_car:
					col_1.append('SECOM Dataset')
					col_1.append('SECOM Dataset')
				elif dataset == dataset_gas:
					col_1.append('Gas Sensor Dataset')
					col_1.append('Gas Sensor Dataset')
				else:
					col_1.append('Absenteeism Dataset')
					col_1.append('Absenteeism Dataset')
				for_mean = round(np.mean(for_scores),2)
				for_stdev = round(np.std(for_scores),2)
				for_best = round(np.amax(for_scores),2)
				for_worst = round(np.amax(for_scores),2)
				sk_mean = round(np.mean(sk_scores),2)
				sk_stdev = round(np.std(sk_scores),2)
				sk_best = round(np.amax(sk_scores),2)
				sk_worst = round(np.amax(sk_scores),2)
				col_2.append('Własna implementacja')
				col_2.append('Sklearn')
				col_3.append(trees_num)
				col_3.append(trees_num)
				col_4.append(threshold)
				col_4.append(threshold)
				col_5.append(for_mean)
				col_6.append(for_stdev)
				col_7.append(for_best)
				col_8.append(for_worst)
				col_5.append(sk_mean)
				col_6.append(sk_stdev)
				col_7.append(sk_best)
				col_8.append(sk_worst)
				print(trees_num)
	header = ['Zbiór danych', 'Implementacja', 'Ilość drzew', 'Próg selekcji', 'Średnia jakość', 'Odchylenie standardowe', 'Najlepsza jakość', 'Najgorsza jakość']
	file_data = np.column_stack((col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8))
	file_data = np.vstack((header, file_data))
	open('tests_res.csv', 'w').close()
	write_data(file_data)

if __name__ == "__main__":
    main()