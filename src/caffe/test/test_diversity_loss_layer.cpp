#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/diversity_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe{

template<typename TypeParam>
class DiversityLossLayerTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	DiversityLossLayerTest()
		: blob_bottom_data_1_(new Blob<Dtype>(10,5, 1, 1)),
		blob_bottom_data_2_(new Blob<Dtype>(10,5, 1, 1)),
		blob_bottom_data_3_(new Blob<Dtype>(10,5, 1, 1)),
		blob_top_loss_(new Blob<Dtype>()){
		// fill the values
		FillerParameter filler_param;
		filler_param.set_value(Dtype(0));
		ConstantFiller<Dtype> filler_1(filler_param);
		filler_1.Fill(this->blob_bottom_data_1_); blob_bottom_vec_.push_back(blob_bottom_data_1_);
		filler_param.set_value(Dtype(1));
		ConstantFiller<Dtype> filler_2(filler_param);
		filler_2.Fill(this->blob_bottom_data_2_); blob_bottom_vec_.push_back(blob_bottom_data_2_);
		filler_param.set_value(Dtype(2));
		ConstantFiller<Dtype> filler_3(filler_param);
		filler_3.Fill(this->blob_bottom_data_3_); blob_bottom_vec_.push_back(blob_bottom_data_3_);
		blob_top_vec_.push_back(blob_top_loss_);
	}
	virtual ~DiversityLossLayerTest(){
		delete blob_bottom_data_1_;
		delete blob_bottom_data_2_;
		delete blob_bottom_data_3_;
		delete blob_top_loss_;
	}

	void TestForward(){
		LayerParameter layer_parameter;
		DiversityLossLayer<Dtype> layer_weight(layer_parameter);
		layer_weight.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss = layer_weight.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype errorMargin = 0.0;
		const Dtype re = -4.0;
		EXPECT_NEAR(loss, re, errorMargin);
	}
	Blob<Dtype>* const blob_bottom_data_1_;
	Blob<Dtype>* const blob_bottom_data_2_;
	Blob<Dtype>* const blob_bottom_data_3_;
	Blob<Dtype>* const blob_top_loss_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DiversityLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(DiversityLossLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(DiversityLossLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_parameter;
	DiversityLossLayer<Dtype> layer(layer_parameter);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}
