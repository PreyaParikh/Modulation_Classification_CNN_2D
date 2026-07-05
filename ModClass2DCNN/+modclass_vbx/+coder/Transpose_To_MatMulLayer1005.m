classdef Transpose_To_MatMulLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        FC1_MatMul_W
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'FC1_MatMul_W'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = modclass_vbx.coder.Transpose_To_MatMulLayer1005(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = modclass_vbx.Transpose_To_MatMulLayer1005(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
            this_ml.FC1_MatMul_W = cgInstance.FC1_MatMul_W;
        end
    end

    methods
        function this = Transpose_To_MatMulLayer1005(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'FC1_MatMul'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = modclass_vbx.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
            this.FC1_MatMul_W = mlInstance.FC1_MatMul_W;
        end

        function [FC1_MatMul] = predict(this, AP1__)
            if isdlarray(AP1__)
                AP1_ = stripdims(AP1__);
            else
                AP1_ = AP1__;
            end
            AP1NumDims = 4;
            AP1 = modclass_vbx.coder.ops.permuteInputVar(AP1_, [4 3 1 2], 4);

            [FC1_MatMul__, FC1_MatMulNumDims__] = Transpose_To_MatMulGraph1010(this, AP1, AP1NumDims, false);
            FC1_MatMul_ = modclass_vbx.coder.ops.permuteOutputVar(FC1_MatMul__, [2 1], 2);

            FC1_MatMul = dlarray(single(FC1_MatMul_), 'CB');
        end

        function [FC1_MatMul, FC1_MatMulNumDims1012] = Transpose_To_MatMulGraph1010(this, AP1, AP1NumDims, Training)

            % Execute the operators:
            % Transpose:
            [perm1010, flatten_FC1_TransposNumDims] = modclass_vbx.coder.ops.prepareTransposeArgs(this.Vars.TransposePerm1011, AP1NumDims);
            if isempty(perm1010)
                flatten_FC1_Transpos = AP1;
            else
                flatten_FC1_Transpos = permute(modclass_vbx.coder.ops.extractIfDlarray(AP1), perm1010);
            end

            % Reshape:
            [shape1011, flatten_FC1_ReshapeNumDims] = modclass_vbx.coder.ops.prepareReshapeArgs(flatten_FC1_Transpos, this.Vars.flatten_FC1_Reshape_, flatten_FC1_TransposNumDims, 0);
            flatten_FC1_Reshape = reshape(flatten_FC1_Transpos, shape1011{:});

            % MatMul:
            [FC1_MatMul, FC1_MatMulNumDims] = modclass_vbx.coder.ops.onnxMatMul(flatten_FC1_Reshape, this.FC1_MatMul_W, flatten_FC1_ReshapeNumDims, this.NumDims.FC1_MatMul_W);

            % Set graph output arguments
            FC1_MatMulNumDims1012 = FC1_MatMulNumDims;

        end

    end

end