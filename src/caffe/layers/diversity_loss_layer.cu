#include <vector>
#include "caffe/layers/diversity_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template<typename Dtype>
void DiversityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int n = bottom[0]->shape()[0];	//样本个数
	int b_num = bottom.size();		//branch个数
	int d = bottom[0]->count();

	Dtype loss = 0.0;
	for (int i = 0; i < b_num - 1; i++){
		for (int j = i + 1; j < b_num; j++){
			Blob<Dtype> diff(bottom[0]->shape());
			caffe_gpu_sub(d, bottom[i]->gpu_data(), bottom[j]->gpu_data(), diff.mutable_gpu_data());
			//caffe_gpu_axpby(d, Dtype(1), diff.gpu_data(), Dtype(1), diff_.mutable_gpu_data() + i*d);
			//caffe_gpu_axpby(d, Dtype(-1), diff.gpu_data(), Dtype(1), diff_.mutable_gpu_data() + j*d);
			caffe_gpu_add(d, diff.gpu_data(), diff_.gpu_data() + i*d, diff_.mutable_gpu_data() + i*d);
			caffe_gpu_sub(d, diff_.gpu_data() + j*d, diff.gpu_data(), diff_.mutable_gpu_data() + j*d);
			caffe_gpu_powx(d, diff.gpu_data(), Dtype(2), diff.mutable_gpu_data());
			Dtype dot = 0.0;
			caffe_gpu_asum(d, diff.gpu_data(),&dot);
			loss += dot;
		}
	}
	if (b_num != 1){
		loss = Dtype(1) - loss / Dtype(b_num * (b_num - 1) * n);
	}
	top[0]->mutable_cpu_data()[0] = loss;
	//caffe_gpu_set(1, Dtype(loss), top[0]->mutable_gpu_data());
}

template<typename Dtype>
void DiversityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	int n = bottom[0]->num();	//样本个数
	int b_num = bottom.size();		//branch个数
	const int d = bottom[0]->count();
	const Dtype* diff = diff_.gpu_data();
	if (b_num != 1){
		const Dtype alpha = Dtype(-1) * Dtype(2)* top[0]->cpu_diff()[0] / Dtype(b_num * (b_num - 1)* n);
		for (int i = 0; i < b_num; i++){
			if (propagate_down[i]){
				caffe_gpu_scale(d,alpha,diff,bottom[i]->mutable_gpu_diff());
				diff += d;
			}
		}
	}

}
INSTANTIATE_LAYER_GPU_FUNCS(DiversityLossLayer);
}
