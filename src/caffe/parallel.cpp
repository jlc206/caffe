#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif

#include <boost/date_time/posix_time/posix_time.hpp>

#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

//for barrier
#include <boost/thread/barrier.hpp>

using caffe::Timer;

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        cudaDeviceProp a, b;
        CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
        CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
        if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        CUDA_CHECK(
            cudaDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

//   Timer timer;
//   timer.Start();

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif

//     timer.Stop();   
//     LOG(INFO)<< "GPU " << solver_->param().device_id() << " " << timer.MilliSeconds() << " MS SPENT IN ON_START";
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

//    LOG(INFO) << "IN ON_GRADIENTS_READY AND I'M GPU " << solver_->param().device_id();

//   Timer timer;
//   timer.Start();

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
    

//       Dtype val;
//       std::cout<<"diff: ";
//       for (int i = 0; i < 5; i++) {
//         cudaMemcpy(&val, &diff_[i], sizeof(Dtype), cudaMemcpyDeviceToHost);
//         std::cout << val << " ";
//    
//       }
//       std::cout << "\n";

  }
#endif

//    timer.Stop();  
//    LOG(INFO)<< "GPU " << solver_->param().device_id() << " " << timer.MilliSeconds() << " MS SPENT IN ON_GRADIENTS_READY";

}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}




//-------------------------------------------------------------


//FOR CHILDREN
template<typename Dtype>
P2CSync<Dtype>::P2CSync(shared_ptr<Solver<Dtype> > root_solver, P2CSync<Dtype>* parent, 
                const SolverParameter& param, shared_ptr<boost::barrier>, shared_ptr<Dtype> big_gradients, int worker)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      initial_iter_(root_solver->iter()),
      worker_num_(worker),
      num_gpus_(parent->num_gpus_),
      solver_(),
      barrier_(),
      big_gradients_() {
#ifndef CPU_ONLY

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  Caffe::set_root_solver(false);
  solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
  Caffe::set_root_solver(true);
  
  barrier_ = parent_->barrier_;
  big_gradients_ = parent_->big_gradients_;
  
  this->configure(solver_.get());
  solver_->add_callback(this);
  
  CUDA_CHECK(cudaMalloc(&device_grads_, (size_ * sizeof(Dtype) * num_gpus_)));

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

//FOR PARENT
template<typename Dtype>
P2CSync<Dtype>::P2CSync(shared_ptr<Solver<Dtype> > root_solver, const SolverParameter& param, int n_gpus)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(),
      initial_iter_(root_solver->iter()),
      worker_num_(0),
      num_gpus_(n_gpus),
      solver_(),
      barrier_(),
      big_gradients_() {
#ifndef CPU_ONLY

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  solver_ = root_solver;
  big_gradients_.reset(new Dtype[(size_ * n_gpus)]);
  barrier_.reset(new boost::barrier(n_gpus));

  this->configure(solver_.get());
  solver_->add_callback(this);
  
  CUDA_CHECK(cudaMalloc(&device_grads_, (size_ * sizeof(Dtype) * n_gpus)));

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}


template<typename Dtype>
P2CSync<Dtype>::~P2CSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));
  CUDA_CHECK(cudaFree(device_grads_));
  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}


template<typename Dtype> 
void P2CSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  if (solver_->param().random_seed() >= 0) {
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2CSync<Dtype>::on_start() {
}

template<typename Dtype>
void P2CSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY

//   Timer timer;
//   timer.Start();
  
  //copy to your spot in global mem
  Dtype* src = diff_;
  Dtype* dst = big_gradients_.get() + (size_ * worker_num_);

//   LOG(INFO) << "diff_ size is: " << sizeof(diff_);
//   LOG(INFO) << "big_gradients_ size is: " << sizeof(big_gradients_);
//   LOG(INFO) << "writing to offset: " << size_ * worker_num_;
  
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToHost, cudaStreamDefault));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  
  //wait until all others are done
  barrier_->wait();
  
  //do calculation
  
  caffe_gpu_set(size_, Dtype(0), diff_); // not sure if necessary
  
  //copy whole thing to "device big gradients" - malloc'ed
  CUDA_CHECK(cudaMemcpyAsync(device_grads_, big_gradients_.get(), size_ * sizeof(Dtype) * num_gpus_,
        cudaMemcpyHostToDevice, cudaStreamDefault)); //destination, source, amt to copy..
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  //sum
  caffe_gpu_add_strided(size_, num_gpus_, device_grads_, diff_); 
  //then do this thing
  caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  

//   if (solver_->param().device_id() == 0) {
//       Dtype val;
//       std::cout<<"diff: ";
//       for (int i = 0; i < 5; i++) {
//         cudaMemcpy(&val, &diff_[i], sizeof(Dtype), cudaMemcpyDeviceToHost);
//         std::cout << val << " ";
//    
//       }
//       std::cout << "\n";
//   }

#endif

//    timer.Stop();  
//    LOG(INFO)<< "GPU " << solver_->param().device_id() << " " << timer.MilliSeconds() << " MS SPENT IN ON_GRADIENTS_READY";

}

template<typename Dtype>
void P2CSync<Dtype>::run(const vector<int>& gpus) { 
  
  SolverParameter param(solver_->param());
  vector<shared_ptr<P2CSync<Dtype> > > syncs(gpus.size());

  for (int i = 1; i < gpus.size(); ++i) { //go to -1 because parent already created
    param.set_device_id(i);
    syncs[i].reset(new P2CSync<Dtype>(solver_, this, param, barrier_, big_gradients_, i));
  }

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}




//-------------------------------------------------------------



INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);
INSTANTIATE_CLASS(P2CSync);

}  // namespace caffe
