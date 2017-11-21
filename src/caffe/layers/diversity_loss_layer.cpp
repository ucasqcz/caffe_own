#include <vector>
#include "caffe/layers/diversity_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{


template<typename Dtype>
void DiversityLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	int n = bottom[0]->num();
	int b = bottom.size();
	int dim = bottom[0]->count(1);
	for (int i = 0; i < bottom.size(); i++){
		CHECK_EQ(bottom[i]->count(1), dim)
			<< "Inputs must have the same dimension.";
	}
	diff_.Reshape(n*b,bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}
/*
template<typename Dtype>
void DiversityLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::Reshape(bottom, top);
	int n = bottom[0]->shape()[0];
	int c = bottom[0]->shape()[1];
	int w = bottom[0]->shape()[2];
	int h = bottom[0]->shape()[3];
	int b = bottom.size();
	diff_.Reshape(n*b, c, w, h);
	std::cout<<n<<" "<<b<<" "<<c<<" "<<c<<" "<<w<<" "<<h<<std::endl;
}
*/
template<typename Dtype>
void DiversityLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	int n = bottom[0]->shape()[0];	//样本个数
	int b_num = bottom.size();		//branch个数
	int d = bottom[0]->count();
	Dtype loss = 0.0;


	for (int i = 0; i < b_num - 1; i++){
		for (int j = i + 1; j < b_num; j++){
			Blob<Dtype> diff(bottom[0]->shape());
			caffe_sub(d, bottom[i]->cpu_data(), bottom[j]->cpu_data(), diff.mutable_cpu_data());
			//caffe_cpu_axpby(d, Dtype(1), diff.cpu_data(), Dtype(1), diff_.mutable_cpu_data() + i*d);
			//caffe_cpu_axpby(d, Dtype(-1), diff.cpu_data(), Dtype(1), diff_.mutable_cpu_data() + j*d);
			caffe_add(d, diff.cpu_data(), diff_.cpu_data() + i*d, diff_.mutable_cpu_data() + i*d);
			caffe_sub(d, diff_.cpu_data() + j*d, diff.cpu_data(), diff_.mutable_cpu_data() + j*d);
			caffe_powx(d, diff.cpu_data(), Dtype(2), diff.mutable_cpu_data());
			Dtype dot = caffe_cpu_asum(d, diff.cpu_data());
			loss += dot;
		}
	}
	if (b_num != 1){ 
		loss = 1 - loss / (b_num * (b_num - 1) * n);
	}
	top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void DiversityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	int n = bottom[0]->num();	//样本个数
	int b_num = bottom.size();		//branch个数
	int d = bottom[0]->count();
	

	if (b_num != 1){
		Dtype alpha = Dtype(-1) * Dtype(2) * top[0]->cpu_diff()[0] / Dtype(b_num * (b_num - 1) * n);
		//CHECK_EQ(1, top[0]->cpu_diff()[0]);
		//CHECK_EQ(-0.033333, alpha);
		for (int i = 0; i < b_num; i++){
			if (propagate_down[i]){
				caffe_cpu_axpby(d, alpha, diff_.cpu_data() + i*d, Dtype(1), bottom[i]->mutable_cpu_diff());
				//caffe_copy(d, tmp.cpu_data() + i*d, bottom[i]->mutable_cpu_diff());
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(DiversityLossLayer)
#endif

INSTANTIATE_CLASS(DiversityLossLayer);
REGISTER_LAYER_CLASS(DiversityLoss);

}
