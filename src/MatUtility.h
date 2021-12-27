#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>

namespace MatUtility
{
    /////////////////////////////// forEach method for use with 2 cv::Mat ////////////////////////////
    template<typename _Tp, typename Functor> inline
    void forEach_2_impl(cv::Mat_<_Tp> *const mat1, cv::Mat_<_Tp> *const mat2, const Functor &operation)
    {
        if (false) {
            operation(*reinterpret_cast<_Tp *>(0), *reinterpret_cast<_Tp *>(0), reinterpret_cast<int *>(0));
            // If your compiler fails in this line.
            // Please check that your functor signature is
            //     (_Tp&, _Tp&, const int*)   <- multi-dimensional
            //  or (_Tp&, _Tp&, void*)        <- in case you don't need current idx.
        }

        CV_Assert(!mat1->empty());
        CV_Assert(!mat2->empty());
        CV_Assert(mat1->size() == mat2->size());
        CV_Assert(mat1->total() / mat1->size[mat1->dims - 1] <= INT_MAX);
        const int LINES = static_cast<int>(mat1->total() / mat1->size[mat1->dims - 1]);

        class MultiPixelOperationWrapper_2 : public cv::ParallelLoopBody
        {
        public:
            MultiPixelOperationWrapper_2(cv::Mat_<_Tp> *const frame1, cv::Mat_<_Tp> *const frame2, const Functor &_operation)
                : mat1(frame1), mat2(frame2), op(_operation) {}
            virtual ~MultiPixelOperationWrapper_2() {}
            // ! Overloaded virtual operator
            // convert range call to row call.
            virtual void operator()(const cv::Range &range) const CV_OVERRIDE
            {
                const int DIMS = mat1->dims;
                const int COLS = mat1->size[DIMS - 1];
                if (DIMS <= 2) {
                    for (int row = range.start; row < range.end; ++row) {
                        this->rowCall2(row, COLS);
                    }
                }
                else {
                    std::vector<int> idx(DIMS); /// idx is modified in this->rowCall
                    idx[DIMS - 2] = range.start - 1;

                    for (int line_num = range.start; line_num < range.end; ++line_num) {
                        idx[DIMS - 2]++;
                        for (int i = DIMS - 2; i >= 0; --i) {
                            if (idx[i] >= mat1->size[i]) {
                                idx[i - 1] += idx[i] / mat1->size[i];
                                idx[i] %= mat1->size[i];
                                continue; // carry-over;
                            }
                            else {
                                break;
                            }
                        }
                        this->rowCall(&idx[0], COLS, DIMS);
                    }
                }
            }
        private:
            cv::Mat_<_Tp> *const mat1, *const mat2;
            const Functor op;
            // ! Call operator for each elements in this row.
            inline void rowCall(int *const idx, const int COLS, const int DIMS) const {
                int &col = idx[DIMS - 1];
                col = 0;
                _Tp *pixel1 = &(mat1->template at<_Tp>(idx));
                _Tp *pixel2 = &(mat2->template at<_Tp>(idx));

                while (col < COLS) {
                    op(*pixel1, *pixel2, const_cast<const int *>(idx));
                    pixel1++; pixel2++; col++;
                }
                col = 0;
            }
            // ! Call operator for each elements in this row. 2d mat special version.
            inline void rowCall2(const int row, const int COLS) const {
                union Index {
                    int body[2];
                    operator const int *() const {
                        return reinterpret_cast<const int *>(this);
                    }
                    int &operator[](const int i) {
                        return body[i];
                    }
                } idx = { {row, 0} };
                // Special union is needed to avoid
                // "error: array subscript is above array bounds [-Werror=array-bounds]"
                // when call the functor `op` such that access idx[3].

                _Tp *pixel1 = &(mat1->template at<_Tp>(idx));
                _Tp *pixel2 = &(mat2->template at<_Tp>(idx));
                const _Tp *const pixel1_end = pixel1 + COLS;
                while (pixel1 < pixel1_end) {
                    op(*pixel1++, *pixel2++, static_cast<const int *>(idx));
                    idx[1]++;
                }
            }
            MultiPixelOperationWrapper_2 &operator=(const MultiPixelOperationWrapper_2 &) {
                CV_Assert(false);
                // We can not remove this implementation because Visual Studio warning C4822.
                return *this;
            }
        };

        cv::parallel_for_(cv::Range(0, LINES), MultiPixelOperationWrapper_2(mat1, mat2, operation));
    }

    /////////////////////////////// forEach method for use with 3 cv::Mat ////////////////////////////
    template<typename _Tp, typename Functor> inline
        void forEach_3_impl(cv::Mat_<_Tp> *const mat1, cv::Mat_<_Tp> *const mat2, cv::Mat_<_Tp> *const mat3, const Functor &operation)
    {
        if (false) {
            operation(*reinterpret_cast<_Tp *>(0), *reinterpret_cast<_Tp *>(0), *reinterpret_cast<_Tp *>(0), reinterpret_cast<int *>(0));
            // If your compiler fails in this line.
            // Please check that your functor signature is
            //     (_Tp&, _Tp&, _Tp&, const int*)   <- multi-dimensional
            //  or (_Tp&, _Tp&, _Tp&, void*)        <- in case you don't need current idx.
        }

        CV_Assert(!mat1->empty());
        CV_Assert(!mat2->empty());
        CV_Assert(!mat3->empty());
        CV_Assert(mat1->size() == mat2->size());
        CV_Assert(mat1->size() == mat3->size());
        CV_Assert(mat1->total() / mat1->size[mat1->dims - 1] <= INT_MAX);
        const int LINES = static_cast<int>(mat1->total() / mat1->size[mat1->dims - 1]);

        class MultiPixelOperationWrapper_3 : public cv::ParallelLoopBody
        {
        public:
            MultiPixelOperationWrapper_3(cv::Mat_<_Tp> *const frame1, cv::Mat_<_Tp> *const frame2, cv::Mat_<_Tp> *const frame3, const Functor &_operation)
                : mat1(frame1), mat2(frame2), mat3(frame3), op(_operation) {}
            virtual ~MultiPixelOperationWrapper_3() {}
            // ! Overloaded virtual operator
            // convert range call to row call.
            virtual void operator()(const cv::Range &range) const CV_OVERRIDE
            {
                const int DIMS = mat1->dims;
                const int COLS = mat1->size[DIMS - 1];
                if (DIMS <= 2) {
                    for (int row = range.start; row < range.end; ++row) {
                        this->rowCall2(row, COLS);
                    }
                }
                else {
                    std::vector<int> idx(DIMS); /// idx is modified in this->rowCall
                    idx[DIMS - 2] = range.start - 1;

                    for (int line_num = range.start; line_num < range.end; ++line_num) {
                        idx[DIMS - 2]++;
                        for (int i = DIMS - 2; i >= 0; --i) {
                            if (idx[i] >= mat1->size[i]) {
                                idx[i - 1] += idx[i] / mat1->size[i];
                                idx[i] %= mat1->size[i];
                                continue; // carry-over;
                            }
                            else {
                                break;
                            }
                        }
                        this->rowCall(&idx[0], COLS, DIMS);
                    }
                }
            }
        private:
            cv::Mat_<_Tp> *const mat1, *const mat2, *const mat3;
            const Functor op;
            // ! Call operator for each elements in this row.
            inline void rowCall(int *const idx, const int COLS, const int DIMS) const {
                int &col = idx[DIMS - 1];
                col = 0;
                _Tp *pixel1 = &(mat1->template at<_Tp>(idx));
                _Tp *pixel2 = &(mat2->template at<_Tp>(idx));
                _Tp *pixel3 = &(mat3->template at<_Tp>(idx));

                while (col < COLS) {
                    op(*pixel1, *pixel2, *pixel3, const_cast<const int *>(idx));
                    pixel1++; pixel2++; pixel3++; col++;
                }
                col = 0;
            }
            // ! Call operator for each elements in this row. 2d mat special version.
            inline void rowCall2(const int row, const int COLS) const {
                union Index {
                    int body[2];
                    operator const int *() const {
                        return reinterpret_cast<const int *>(this);
                    }
                    int &operator[](const int i) {
                        return body[i];
                    }
                } idx = { {row, 0} };
                // Special union is needed to avoid
                // "error: array subscript is above array bounds [-Werror=array-bounds]"
                // when call the functor `op` such that access idx[3].

                _Tp *pixel1 = &(mat1->template at<_Tp>(idx));
                _Tp *pixel2 = &(mat2->template at<_Tp>(idx));
                _Tp *pixel3 = &(mat3->template at<_Tp>(idx));
                const _Tp *const pixel1_end = pixel1 + COLS;
                while (pixel1 < pixel1_end) {
                    op(*pixel1++, *pixel2++, *pixel3++, static_cast<const int *>(idx));
                    idx[1]++;
                }
            }
            MultiPixelOperationWrapper_3 &operator=(const MultiPixelOperationWrapper_3 &) {
                CV_Assert(false);
                // We can not remove this implementation because Visual Studio warning C4822.
                return *this;
            }
        };

        cv::parallel_for_(cv::Range(0, LINES), MultiPixelOperationWrapper_3(mat1, mat2, mat3, operation));
    }
};

