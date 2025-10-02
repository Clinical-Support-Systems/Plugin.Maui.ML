using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MauiSample.Models
{
    /// <summary>
    /// Represents a medical entity extracted from text
    /// </summary>
    public class MedicalEntity
    {
        public string Text { get; set; } = string.Empty;
        public string EntityType { get; set; } = string.Empty; // e.g., "MEDICATION", "DIAGNOSIS", "PROCEDURE"
        public float Confidence { get; set; }
        public int StartPosition { get; set; }
        public int EndPosition { get; set; }
    }
}
