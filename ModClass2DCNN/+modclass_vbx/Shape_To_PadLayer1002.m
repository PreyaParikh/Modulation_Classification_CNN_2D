classdef Shape_To_PadLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN3_Split_split
        CNN3_Pad_value
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
            name = 'modclass_vbx.coder.Shape_To_PadLayer1002';
        end
    end


    methods
        function this = Shape_To_PadLayer1002(name)
            this.Name = name;
            this.OutputNames = {'CNN3_Pad'};
        end

        function [CNN3_Pad] = predict(this, MaxPool2)
            if isdlarray(MaxPool2)
                MaxPool2 = stripdims(MaxPool2);
            end
            MaxPool2NumDims = 4;
            MaxPool2 = modclass_vbx.ops.permuteInputVar(MaxPool2, [4 3 1 2], 4);

            [CNN3_Pad, CNN3_PadNumDims] = Shape_To_PadGraph1004(this, MaxPool2, MaxPool2NumDims, false);
            CNN3_Pad = modclass_vbx.ops.permuteOutputVar(CNN3_Pad, [3 4 2 1], 4);

            CNN3_Pad = dlarray(single(CNN3_Pad), 'SSCB');
        end

        function [CNN3_Pad] = forward(this, MaxPool2)
            if isdlarray(MaxPool2)
                MaxPool2 = stripdims(MaxPool2);
            end
            MaxPool2NumDims = 4;
            MaxPool2 = modclass_vbx.ops.permuteInputVar(MaxPool2, [4 3 1 2], 4);

            [CNN3_Pad, CNN3_PadNumDims] = Shape_To_PadGraph1004(this, MaxPool2, MaxPool2NumDims, true);
            CNN3_Pad = modclass_vbx.ops.permuteOutputVar(CNN3_Pad, [3 4 2 1], 4);

            CNN3_Pad = dlarray(single(CNN3_Pad), 'SSCB');
        end

        function [CNN3_Pad, CNN3_PadNumDims1005] = Shape_To_PadGraph1004(this, MaxPool2, MaxPool2NumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN3_InputShapeInt, CNN3_InputShapeIntNumDims] = modclass_vbx.ops.onnxShape(MaxPool2, MaxPool2NumDims, 0, MaxPool2NumDims+1);

            % Cast:
            CNN3_InputShape = single(CNN3_InputShapeInt);
            CNN3_InputShapeNumDims = CNN3_InputShapeIntNumDims;

            % Div:
            CNN3_Div = CNN3_InputShape ./ this.Vars.CNN3_Stride;
            CNN3_DivNumDims = max(CNN3_InputShapeNumDims, this.NumDims.CNN3_Stride);

            % Ceil:
            CNN3_Ceil = ceil(CNN3_Div);
            CNN3_CeilNumDims = CNN3_DivNumDims;

            % Sub:
            CNN3_Sub = CNN3_Ceil - this.Vars.CNN3_One;
            CNN3_SubNumDims = max(CNN3_CeilNumDims, this.NumDims.CNN3_One);

            % Mul:
            CNN3_Mul = CNN3_Sub .* this.Vars.CNN3_Stride;
            CNN3_MulNumDims = max(CNN3_SubNumDims, this.NumDims.CNN3_Stride);

            % Add:
            CNN3_Add = CNN3_Mul + this.Vars.CNN3_Filter;
            CNN3_AddNumDims = max(CNN3_MulNumDims, this.NumDims.CNN3_Filter);

            % Sub:
            CNN3_Sub1 = CNN3_Add - CNN3_InputShape;
            CNN3_Sub1NumDims = max(CNN3_AddNumDims, CNN3_InputShapeNumDims);

            % Relu:
            CNN3_ReluInt = relu(dlarray(CNN3_Sub1));
            CNN3_ReluIntNumDims = CNN3_Sub1NumDims;

            % Cast:
            CNN3_Relu = single(CNN3_ReluInt);
            CNN3_ReluNumDims = CNN3_ReluIntNumDims;

            % Split:
            [CNN3_BCPadding, CNN3_totalPaddingNee, CNN3_totalPaddingN_5, CNN3_BCPaddingNumDims, CNN3_totalPaddingNeeNumDims, CNN3_totalPaddingN_5NumDims] = modclass_vbx.ops.onnxSplit13(CNN3_Relu, 0, this.CNN3_Split_split, 3, CNN3_ReluNumDims);

            % Div:
            CNN3_totalPaddingN_2 = CNN3_totalPaddingNee ./ this.Vars.CNN3_totalPaddingN_4;
            CNN3_totalPaddingN_2NumDims = max(CNN3_totalPaddingNeeNumDims, this.NumDims.CNN3_totalPaddingN_4);

            % Floor:
            CNN3_totalPaddingN_3 = floor(CNN3_totalPaddingN_2);
            CNN3_totalPaddingN_3NumDims = CNN3_totalPaddingN_2NumDims;

            % Ceil:
            CNN3_totalPaddingN_1 = ceil(CNN3_totalPaddingN_2);
            CNN3_totalPaddingN_1NumDims = CNN3_totalPaddingN_2NumDims;

            % Div:
            CNN3_totalPaddingN_7 = CNN3_totalPaddingN_5 ./ this.Vars.CNN3_totalPaddingN_9;
            CNN3_totalPaddingN_7NumDims = max(CNN3_totalPaddingN_5NumDims, this.NumDims.CNN3_totalPaddingN_9);

            % Floor:
            CNN3_totalPaddingN_8 = floor(CNN3_totalPaddingN_7);
            CNN3_totalPaddingN_8NumDims = CNN3_totalPaddingN_7NumDims;

            % Ceil:
            CNN3_totalPaddingN_6 = ceil(CNN3_totalPaddingN_7);
            CNN3_totalPaddingN_6NumDims = CNN3_totalPaddingN_7NumDims;

            % Concat:
            [CNN3_paddingSize, CNN3_paddingSizeNumDims] = modclass_vbx.ops.onnxConcat(0, {this.Vars.CNN3_Zero, this.Vars.CNN3_Zero, CNN3_totalPaddingN_3, CNN3_totalPaddingN_8, this.Vars.CNN3_Zero, this.Vars.CNN3_Zero, CNN3_totalPaddingN_1, CNN3_totalPaddingN_6}, [this.NumDims.CNN3_Zero, this.NumDims.CNN3_Zero, CNN3_totalPaddingN_3NumDims, CNN3_totalPaddingN_8NumDims, this.NumDims.CNN3_Zero, this.NumDims.CNN3_Zero, CNN3_totalPaddingN_1NumDims, CNN3_totalPaddingN_6NumDims]);

            % Cast:
            CNN3_paddingSizeInt = cast(int64(extractdata(CNN3_paddingSize)), 'like', CNN3_paddingSize);
            CNN3_paddingSizeIntNumDims = CNN3_paddingSizeNumDims;

            % Pad:
            [CNN3_Pad, CNN3_PadNumDims] = modclass_vbx.ops.onnxPad(MaxPool2, CNN3_paddingSizeInt, this.CNN3_Pad_value, 'constant', dlarray([0:MaxPool2NumDims]'), MaxPool2NumDims);

            % Set graph output arguments
            CNN3_PadNumDims1005 = CNN3_PadNumDims;

        end

    end

end