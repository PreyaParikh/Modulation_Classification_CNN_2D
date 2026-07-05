classdef Transpose_To_MatMulLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

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
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'modclass_vbx.coder.Transpose_To_MatMulLayer1005';
        end
    end


    methods
        function this = Transpose_To_MatMulLayer1005(name)
            this.Name = name;
            this.OutputNames = {'FC1_MatMul'};
        end

        function [FC1_MatMul] = predict(this, AP1)
            if isdlarray(AP1)
                AP1 = stripdims(AP1);
            end
            AP1NumDims = 4;
            AP1 = modclass_vbx.ops.permuteInputVar(AP1, [4 3 1 2], 4);

            [FC1_MatMul, FC1_MatMulNumDims] = Transpose_To_MatMulGraph1010(this, AP1, AP1NumDims, false);
            FC1_MatMul = modclass_vbx.ops.permuteOutputVar(FC1_MatMul, [2 1], 2);

            FC1_MatMul = dlarray(single(FC1_MatMul), 'CB');
        end

        function [FC1_MatMul] = forward(this, AP1)
            if isdlarray(AP1)
                AP1 = stripdims(AP1);
            end
            AP1NumDims = 4;
            AP1 = modclass_vbx.ops.permuteInputVar(AP1, [4 3 1 2], 4);

            [FC1_MatMul, FC1_MatMulNumDims] = Transpose_To_MatMulGraph1010(this, AP1, AP1NumDims, true);
            FC1_MatMul = modclass_vbx.ops.permuteOutputVar(FC1_MatMul, [2 1], 2);

            FC1_MatMul = dlarray(single(FC1_MatMul), 'CB');
        end

        function [FC1_MatMul, FC1_MatMulNumDims1012] = Transpose_To_MatMulGraph1010(this, AP1, AP1NumDims, Training)

            % Execute the operators:
            % Transpose:
            [perm, flatten_FC1_TransposNumDims] = modclass_vbx.ops.prepareTransposeArgs(this.Vars.TransposePerm1011, AP1NumDims);
            if isempty(perm)
                flatten_FC1_Transpos = AP1;
            else
                flatten_FC1_Transpos = permute(AP1, perm);
            end

            % Reshape:
            [shape, flatten_FC1_ReshapeNumDims] = modclass_vbx.ops.prepareReshapeArgs(flatten_FC1_Transpos, this.Vars.flatten_FC1_Reshape_, flatten_FC1_TransposNumDims, 0);
            flatten_FC1_Reshape = reshape(flatten_FC1_Transpos, shape{:});

            % MatMul:
            [FC1_MatMul, FC1_MatMulNumDims] = modclass_vbx.ops.onnxMatMul(flatten_FC1_Reshape, this.FC1_MatMul_W, flatten_FC1_ReshapeNumDims, this.NumDims.FC1_MatMul_W);

            % Set graph output arguments
            FC1_MatMulNumDims1012 = FC1_MatMulNumDims;

        end

    end

end