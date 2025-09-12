using System.Windows.Input;
using MauiSample.Models;

namespace MauiSample.Pages.Controls
{
    public partial class TaskView
    {
        public static readonly BindableProperty TaskCompletedCommandProperty = BindableProperty.Create(
            nameof(TaskCompletedCommand),
            typeof(ICommand),
            typeof(TaskView));

        public TaskView()
        {
            InitializeComponent();
        }

        public ICommand TaskCompletedCommand
        {
            get => (ICommand)GetValue(TaskCompletedCommandProperty);
            set => SetValue(TaskCompletedCommandProperty, value);
        }

        private void CheckBox_CheckedChanged(object sender, CheckedChangedEventArgs e)
        {
            var checkbox = (CheckBox)sender;

            if (checkbox.BindingContext is not ProjectTask task)
                return;

            if (task.IsCompleted == e.Value)
                return;

            task.IsCompleted = e.Value;
            TaskCompletedCommand?.Execute(task);
        }
    }
}
