import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("СХОПЫВАНИЕ ПРЕДСКАЗАНИЙ 10-КЛАССОВОГО КЛАССИФИКАТОРА")

results_file = r"C:\UIRS\surface-classification\classification_results_dct.csv"
print(f"\nЗагрузка результатов из: {results_file}")
df_results = pd.read_csv(results_file)
print(f"Загружено {len(df_results):,} предсказаний")

true_labels = df_results['true_surface'].values
predicted_labels = df_results['predicted_labels'].values

# Исходная точность на 10 классах
accuracy_10 = accuracy_score(true_labels, predicted_labels)
print(f"\nИсходная точность (10 классов): {accuracy_10*100:.2f}%")


mappings = {
    'Статья (3 класса)': {
        'long_carpet': 'Lateral_Slip', 'artificial_grass': 'Lateral_Slip',
        'laminate_flooring': 'Stable', 'short_carpet': 'Stable', 
        'linoleum': 'Stable', 'ceramic_tiles': 'Stable',
        'pvc_foamboard': 'Longitudinal_Slip', 'osb': 'Longitudinal_Slip', 
        'foam_underlayment': 'Longitudinal_Slip', 'eva_foam_tiles': 'Longitudinal_Slip'
    },
    '4 класса': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 
        'artificial_grass': 'Soft_Rough', 'osb': 'Soft_Rough', 
        'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 
        'linoleum': 'Hard_Floors',
        'laminate_flooring': 'Laminate'
    },
    '6 классов': {
        'long_carpet': 'Soft_Grass', 'short_carpet': 'Soft_Grass', 
        'artificial_grass': 'Soft_Grass',
        'osb': 'OSB_Foam', 'foam_underlayment': 'OSB_Foam',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
        'linoleum': 'Linoleum', 'laminate_flooring': 'Laminate'
    },
    'Объединение EVA+Hard (3 класса)': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 
        'artificial_grass': 'Soft_Rough', 'osb': 'Soft_Rough', 
        'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'Hard_EVA_Merged', 
        'pvc_foamboard': 'Hard_EVA_Merged', 'ceramic_tiles': 'Hard_EVA_Merged', 
        'linoleum': 'Hard_EVA_Merged',
        'laminate_flooring': 'Laminate'
    }
}

# Точности обученных моделей (для сравнения)
trained_accuracies = {
    'Статья (3 класса)': 0.457,
    '4 класса': 0.747,
    '6 классов': 0.626,
    'Объединение EVA+Hard (3 класса)': 0.797
}

results_summary = []

for method_name, mapping in mappings.items():
    print(f"МЕТОД: {method_name}")
    
    # Применяем маппинг
    true_merged = pd.Series(true_labels).map(mapping)
    predicted_merged = pd.Series(predicted_labels).map(mapping)
    
    # Accuracy после схлопывания
    accuracy_merged = accuracy_score(true_merged, predicted_merged)
    gain = accuracy_merged - accuracy_10
    
    print(f"\nТочность исходная (10 классов): {accuracy_10*100:.2f}%")
    print(f"Точность после схлопывания:     {accuracy_merged*100:.2f}%")
    print(f"Прирост:                         {gain*100:+.2f}%")
    
    # Сравнение с обученной моделью
    if method_name in trained_accuracies:
        trained_acc = trained_accuracies[method_name]
        diff = accuracy_merged - trained_acc
        print(f"\nТочность обученной модели:       {trained_acc*100:.2f}%")
        print(f"Разница (схлопывание - обучение): {diff*100:+.2f}%")
    

    classes = sorted(list(set(true_merged.unique()) | set(predicted_merged.unique())))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_merged, predicted_merged, labels=classes, zero_division=0
    )
    
    print(f"\nClassification Report:")
    print(f"{'Класс':<25} {'Precision':<12} {'Recall':<12} {'Support':<10}")

    for i, cls in enumerate(classes):
        print(f"{str(cls):<25} {precision[i]:<12.3f} {recall[i]:<12.3f} {support[i]:<10}")
    
    print(f"{'Accuracy':<25} {accuracy_merged:<12.3f}")
    print(f"{'Macro avg':<25} {np.mean(precision):<12.3f} {np.mean(recall):<12.3f}")
    print(f"{'Weighted avg':<25} {np.average(precision, weights=support):<12.3f} {np.average(recall, weights=support):<12.3f}")
    
    results_summary.append({
        'Метод': method_name,
        'Кол-во классов': len(set(mapping.values())),
        'Accuracy исходная': f"{accuracy_10*100:.2f}%",
        'Accuracy после схлопывания': f"{accuracy_merged*100:.2f}%",
        'Прирост': f"{gain*100:+.2f}%",
        'Accuracy обученной модели': f"{trained_accuracies.get(method_name, 'N/A')*100:.2f}%" if method_name in trained_accuracies else 'N/A'
    })


print("ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")

df_summary = pd.DataFrame(results_summary)
print(df_summary.to_string(index=False))

# Сохраняем таблицу
summary_file = r"C:\UIRS\surface-classification\merge_comparison_summary.csv"
df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"\nТаблица сохранена: {summary_file}")